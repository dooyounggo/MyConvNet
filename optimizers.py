"""
Define optimization process.
Includes gradient descent, weight decay, TensorBoard summaries, learning rate updates, validation, etc.
"""

import os
import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from utils import plot_learning_curve
from tensorflow.python.client import timeline
from tensorflow.python import pywrap_tensorflow


class Optimizer(object):
    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer.
        :param model: ConvNet, the model to be trained.
        :param train_set: DataSet, training set to be used.
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: DataSet, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra arguments containing hyperparameters.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        self.input_size = kwargs.get('input_size', [224, 224, 3])
        assert len(self.input_size) == 3, 'input_size must contain 3D size'

        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.monte_carlo = kwargs.get('monte_carlo', False)
        self.augment_train = kwargs.get('augment_train', False)
        self.init_learning_rate = kwargs.get('base_learning_rate', 0.1)*self.batch_size/256

        self.warmup_epoch = kwargs.get('learning_warmup_epoch', 0)
        self.decay_method = kwargs.get('learning_rate_decay_method', 'cosine')
        self.decay_params = kwargs.get('learning_rate_decay_params', (0.94, 2))

        self.update_vars = tf.trainable_variables()
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('calc/'):
            self.learning_rate_multiplier = tf.placeholder(dtype=tf.float32, name='learning_rate_multiplier')
            self.learning_rate = self.init_learning_rate*self.learning_rate_multiplier

        self.optimization_operation = self._optimize_and_update(self._optimizer(**kwargs), **kwargs)

        self._reset()

    def _reset(self):
        self.curr_step = 0
        self.curr_epoch = 1
        self.best_score = self.evaluator.worst_score
        self.learning_rate_update = 0
        self.curr_multiplier = 1.0

    @abstractmethod
    def _optimizer(self, **kwargs):
        """
        tf.train.Optimizer for a gradient update
        This should be implemented, and should not be called manually.
        """
        pass

    def _optimize_and_update(self, optimizer, **kwargs):
        gradient_threshold = kwargs.get('gradient_threshold', None)
        loss_scaling_factor = kwargs.get('loss_scaling_factor', 1.0)
        weight_decay = kwargs.get('base_weight_decay', 0.0)*self.batch_size/256
        weight_decay_scheduling = kwargs.get('weight_decay_scheduling', True)
        l1_weight_decay = kwargs.get('l1_weight_decay', False)
        huber_decay_delta = kwargs.get('huber_decay_delta', None)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.model.gpu_offset, self.model.num_gpus + self.model.gpu_offset):
                with tf.device('/gpu:' + str(i)):
                    with tf.variable_scope('gpu{}/gradients'.format(i)):
                        loss = self.model.losses[i - self.model.gpu_offset]
                        if loss_scaling_factor > 1.0:
                            loss *= loss_scaling_factor
                        if self.model.dtype is not tf.float32:
                            loss = tf.cast(loss, dtype=self.model.dtype)
                        grads_and_vars = optimizer.compute_gradients(loss, var_list=self.update_vars)
                        grads, gvars = zip(*grads_and_vars)
                        grads = list(grads)
                        if loss_scaling_factor > 1.0:
                            for ng in range(len(grads)):
                                grads[ng] /= loss_scaling_factor
                        if gradient_threshold is not None:
                            grads, _ = tf.clip_by_global_norm(grads, gradient_threshold)
                        tower_grads.append([gv for gv in zip(grads, gvars)])
                        tf.get_variable_scope().reuse_variables()

        if self.model.num_gpus == 1:
            avg_grads_and_vars = tower_grads[0]
            self.avg_grads = grads
        else:
            with tf.device(self.model.param_device):
                with tf.variable_scope('calc/mean_gradients'):
                    avg_grads = []
                    avg_vars = []
                    for grads_and_vars in zip(*tower_grads):
                        # Note that each grads_and_vars looks like the following:
                        # ( (grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN) )
                        grads = []
                        for g, _ in grads_and_vars:
                            # g = tf.where(tf.is_nan(g), tf.zeros_like(g), g)  # Prevent NaNs
                            # if self.model.dtype is not tf.float32:
                            #     g = tf.cast(g, dtype=tf.float32)
                            g_exp = tf.expand_dims(g, 0)
                            # Append on a 'tower' dimension which we will average over below.
                            grads.append(g_exp)

                        grad = tf.concat(grads, axis=0)
                        grad = tf.reduce_mean(grad, axis=0)

                        # Pointers to the variables are the same for all towers since the variables are shared.
                        avg_vars.append(grads_and_vars[0][1])
                        avg_grads.append(grad)
                    # if gradient_threshold is not None:
                    #     avg_grads, _ = tf.clip_by_global_norm(avg_grads, gradient_threshold)

                    avg_grads_and_vars = [gv for gv in zip(avg_grads, avg_vars)]
                    self.avg_grads = avg_grads

        decay_ops = []
        if weight_decay > 0.0:
            variables = tf.get_collection('weight_variables')
            if kwargs.get('bias_norm_decay', False):
                variables += tf.get_collection('bias_variables') + tf.get_collection('norm_variables')
            with tf.variable_scope('weight_decay'):
                weight_decay = tf.constant(weight_decay, dtype=tf.float32, name='weight_decay_factor')
                if weight_decay_scheduling:
                    weight_decay = self.learning_rate_multiplier*weight_decay
                if huber_decay_delta is not None:
                    delta = tf.constant(huber_decay_delta, dtype=tf.float32, name='huber_delta')
                for var in variables:
                    if var.trainable:
                        if huber_decay_delta is None:
                            if l1_weight_decay:
                                decay_op = var.assign_sub(weight_decay*tf.math.sign(var))
                            else:
                                decay_op = var.assign_sub(weight_decay*var)
                        else:  # Pseudo-Huber weight decay
                            decay_op = var.assign_sub(weight_decay*var/tf.math.sqrt(1 + (var/delta)**2))
                        decay_ops.append(decay_op)
        with tf.control_dependencies(self.model.update_ops):
            with tf.control_dependencies([optimizer.apply_gradients(avg_grads_and_vars,
                                                                    global_step=self.model.global_step)]
                                         + self.update_ops):
                opt_op = tf.group(decay_ops)
        return opt_op

    def train(self, save_dir='./tmp', transfer_dir=None, details=False, verbose=True, show_each_step=False, **kwargs):
        train_size = self.train_set.num_examples
        num_steps_per_epoch = np.ceil(train_size/self.batch_size).astype(int)
        self.steps_per_epoch = num_steps_per_epoch
        num_steps = num_steps_per_epoch*self.num_epochs
        self.total_steps = num_steps

        validation_frequency = kwargs.get('validation_frequency', None)
        summary_frequency = kwargs.get('summary_frequency', None)
        if validation_frequency is None:
            validation_frequency = num_steps_per_epoch
        if summary_frequency is None:
            summary_frequency = num_steps_per_epoch

        num_validations = num_steps//validation_frequency
        last_val_iter = num_validations*validation_frequency

        if transfer_dir is not None:  # Transfer learning setup
            model_to_load = kwargs.get('model_to_load', None)
            blocks_to_load = kwargs.get('blocks_to_load', None)
            load_moving_average = kwargs.get('load_moving_average', False)
            start_epoch = kwargs.get('start_epoch', 0)
            start_step = num_steps_per_epoch*start_epoch

            if not os.path.isdir(transfer_dir):
                ckpt_to_load = transfer_dir
            elif model_to_load is None:  # Find a model to be transferred
                ckpt_to_load = tf.train.latest_checkpoint(transfer_dir)
            elif isinstance(model_to_load, str):
                ckpt_to_load = os.path.join(transfer_dir, model_to_load)
            else:
                fp = open(os.path.join(transfer_dir, 'checkpoints.txt'), 'r')
                ckpt_list = fp.readlines()
                fp.close()
                ckpt_to_load = os.path.join(transfer_dir, ckpt_list[model_to_load].rstrip())

            reader = pywrap_tensorflow.NewCheckpointReader(ckpt_to_load)  # Find variables to be transferred
            var_to_shape_map = reader.get_variable_to_shape_map()
            var_names = [var for var in var_to_shape_map.keys()]

            var_list = []
            if blocks_to_load is None:
                for i in range(self.model.num_blocks):
                    var_list += tf.get_collection('block_{}_variables'.format(i))
                    var_list += tf.get_collection('block_{}_ema_variables'.format(i))
                var_list += tf.get_collection('block_{}_variables'.format(None))
                var_list += tf.get_collection('block_{}_ema_variables'.format(None))
            else:
                for i in blocks_to_load:
                    var_list += tf.get_collection('block_{}_variables'.format(i))
                    var_list += tf.get_collection('block_{}_ema_variables'.format(i))

            variables_not_loaded = []
            if load_moving_average:
                variables = {}
                for var in var_list:
                    var_name = var.name.rstrip(':0')
                    ema_name = var.name.rstrip(':0') + '/ExponentialMovingAverage'
                    if ema_name in var_to_shape_map:
                        if var.get_shape() == var_to_shape_map[ema_name]:
                            variables[ema_name] = var
                            if var_name in var_names:
                                var_names.remove(ema_name)
                        else:
                            print('<{}> was not loaded due to shape mismatch'.format(var_name))
                            variables_not_loaded.append(var_name)
                    elif var_name in var_to_shape_map:
                        if var.get_shape() == var_to_shape_map[var_name]:
                            variables[var_name] = var
                            if var_name in var_names:
                                var_names.remove(var_name)
                        else:
                            print('<{}> was not loaded due to shape mismatch'.format(var_name))
                            variables_not_loaded.append(var_name)
                    else:
                        variables_not_loaded.append(var_name)
            else:
                variables = []
                for var in var_list:
                    var_name = var.name.rstrip(':0')
                    if var_name in var_to_shape_map:
                        if var.get_shape() == var_to_shape_map[var_name]:
                            variables.append(var)
                            var_names.remove(var_name)
                        else:
                            print('<{}> was not loaded due to shape mismatch'.format(var_name))
                            variables_not_loaded.append(var_name)
                    else:
                        variables_not_loaded.append(var_name)

            saver_transfer = tf.train.Saver(variables)

            self.model.session.run(tf.global_variables_initializer())
            saver_transfer.restore(self.model.session, ckpt_to_load)

            print('')
            print('Variables have been initialized using the following checkpoint:')
            print(ckpt_to_load)
            print('The following variables in the checkpoint were not used:')
            print(var_names)
            print('The following variables do not exist in the checkpoint, so they were initialized randomly:')
            print(variables_not_loaded)
            print('')

            pkl_file = os.path.join(transfer_dir, 'learning_curve-result-1.pkl')
            pkl_loaded = False
            if os.path.exists(pkl_file):
                train_steps = start_step if show_each_step else start_step//validation_frequency
                eval_steps = start_step//validation_frequency
                with open(pkl_file, 'rb') as fo:
                    prev_results = pkl.load(fo)
                prev_results[0] = prev_results[0][:train_steps]
                prev_results[1] = prev_results[1][:train_steps]
                prev_results[2] = prev_results[2][:eval_steps]
                prev_results[3] = prev_results[3][:eval_steps]
                train_len = len(prev_results[0])
                eval_len = len(prev_results[2])
                if train_len == train_steps and eval_len == eval_steps:
                    train_losses, train_scores, eval_losses, eval_scores = prev_results
                    pkl_loaded = True
                else:
                    train_losses, train_scores, eval_losses, eval_scores = [], [], [], []
            else:
                train_losses, train_scores, eval_losses, eval_scores = [], [], [], []
        else:
            start_epoch = 0
            start_step = 0
            self.model.session.run(tf.global_variables_initializer())
            train_losses, train_scores, eval_losses, eval_scores = [], [], [], []
            pkl_loaded = False

        max_to_keep = kwargs.get('max_to_keep', 5)
        log_trace = kwargs.get('log_trace', False)
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver.export_meta_graph(filename=os.path.join(save_dir, 'model.ckpt.meta'))

        kwargs['monte_carlo'] = False  # Turn off monte carlo dropout for validation

        with tf.device('/cpu:{}'.format(self.model.cpu_offset)):
            with tf.variable_scope('summaries'):  # TensorBoard summaries
                tf.summary.scalar('Loss', self.model.loss)
                tf.summary.scalar('Learning Rate', self.learning_rate)
                tf.summary.scalar('Debug Value', self.model.debug_value)
                tf.summary.image('Input Images',
                                 tf.cast(self.model.input_images*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.image('Augmented Input Images',
                                 tf.cast(self.model.X_all*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.image('Debug Images 0',
                                 tf.cast(self.model.debug_images_0*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.image('Debug Images 1',
                                 tf.cast(self.model.debug_images_1*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.histogram('Image Histogram', self.model.X_all)
                for i in range(self.model.num_blocks):
                    weights = tf.get_collection('block_{}_weight_variables'.format(i))
                    if len(weights) > 0:
                        tf.summary.histogram('Block {} Weight Histogram'.format(i), weights[0])
                weights = tf.get_collection('weight_variables')
                with tf.variable_scope('weights_l1'):
                    weights_l1 = tf.math.accumulate_n([tf.reduce_sum(tf.math.abs(w)) for w in weights])
                    tf.summary.scalar('Weights L1 Norm', weights_l1)
                with tf.variable_scope('weights_l2'):
                    weights_l2 = tf.global_norm(weights)
                    tf.summary.scalar('Weights L2 Norm', weights_l2)
                tail_scores_5 = []
                tail_scores_1 = []
                with tf.variable_scope('weights_tail_score'):
                    for w in weights:
                        w_size = tf.size(w, out_type=tf.float32)
                        w_std = tf.math.reduce_std(w)
                        w_abs = tf.math.abs(w)
                        tail_threshold_5 = 1.96*w_std
                        tail_threshold_1 = 2.58*w_std
                        num_weights_5 = tf.math.reduce_sum(tf.cast(tf.math.greater(w_abs, tail_threshold_5),
                                                                   dtype=tf.float32))
                        num_weights_1 = tf.math.reduce_sum(tf.cast(tf.math.greater(w_abs, tail_threshold_1),
                                                                   dtype=tf.float32))
                        tail_scores_5.append(num_weights_5/(0.05*w_size))
                        tail_scores_1.append(num_weights_1/(0.01*w_size))
                    tail_score_5 = tf.math.accumulate_n(tail_scores_5)/len(tail_scores_5)
                    tail_score_1 = tf.math.accumulate_n(tail_scores_1)/len(tail_scores_1)
                    tf.summary.scalar('Weights Tail Score 5p', tail_score_5)
                    tf.summary.scalar('Weights Tail Score 1p', tail_score_1)
                with tf.variable_scope('gradients_l2'):
                    gradients_l2 = tf.global_norm(self.avg_grads)
                    tf.summary.scalar('Gradients L2 Norm', gradients_l2)
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'logs'), self.model.session.graph)

        train_results = dict()

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))
            print('Number of iterations per epoch: {}'.format(num_steps_per_epoch))

        if show_each_step:
            step_losses, step_scores = [], []
        else:
            step_losses, step_scores = 0, 0
        eval_loss, eval_score = np.inf, 0
        annotations = []

        self.train_set.initialize(self.model.session)  # Initialize training iterator
        handles = self.train_set.get_string_handles(self.model.session)  # Get a string handle from training iterator
        if self.val_set is not None:
            self.val_set.initialize(self.model.session)  # Initialize validation iterator
        with tf.variable_scope('calc/'):
            step_init_op = self.model.global_step.assign(start_step, name='init_global_step')
        self.model.session.run(step_init_op)
        tf.get_default_graph().finalize()

        # self._test_drive(save_dir=save_dir)  # Run test code

        self.curr_epoch += start_epoch
        self.curr_step += start_step
        step_loss, step_score = 0, 0
        start_time = time.time()
        for i in range(num_steps - start_step):  # Training iterations
            self._update_learning_rate()

            try:
                step_loss, step_Y_true, step_Y_pred = self._step(handles, merged=merged, writer=train_writer,
                                                                 summary=i % summary_frequency == 0,
                                                                 log_trace=log_trace and i % summary_frequency == 1)
                step_score = self.evaluator.score(step_Y_true, step_Y_pred)
            except tf.errors.OutOfRangeError:
                if verbose:
                    remainder_size = train_size - (self.steps_per_epoch - 1)*self.batch_size
                    print('The last iteration ({} data) has been ignored'.format(remainder_size))

            if show_each_step:
                step_losses.append(step_loss)
                step_scores.append(step_score)
            else:
                step_losses += step_loss
                step_scores += step_score
            self.curr_step += 1

            if (i + 1) % validation_frequency == 0:  # Validation every validation_frequency iterations
                if self.val_set is not None:
                    _, eval_Y_true, eval_Y_pred, eval_loss = self.model.predict(self.val_set, verbose=False,
                                                                                return_images=False, **kwargs)
                    eval_score = self.evaluator.score(eval_Y_true, eval_Y_pred)
                    eval_scores.append(eval_score)
                    eval_losses.append(eval_loss)

                    del eval_Y_true, eval_Y_pred

                    curr_score = eval_score
                else:
                    curr_score = step_score

                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):  # Save best model
                    self.best_score = curr_score
                    saver.save(self.model.session, os.path.join(save_dir, 'model.ckpt'),
                               global_step=self.model.global_step,
                               write_meta_graph=False)

                    if show_each_step:
                        annotations.append((self.curr_step, curr_score))
                    else:
                        annotations.append((self.curr_step//validation_frequency, curr_score))
                    annotations = annotations[-max_to_keep:]
                elif self.curr_step == last_val_iter:  # Save latest model
                    saver.save(self.model.session, os.path.join(save_dir, 'model.ckpt'),
                               global_step=self.model.global_step,
                               write_meta_graph=False)

                    if show_each_step:
                        annotations.append((self.curr_step, curr_score))
                    else:
                        annotations.append((self.curr_step//validation_frequency, curr_score))
                    annotations = annotations[-max_to_keep:]

                ckpt_list = saver.last_checkpoints[::-1]
                fp = open(os.path.join(save_dir, 'checkpoints.txt'), 'w')
                for fname in ckpt_list:
                    fp.write(fname.split(os.sep)[-1] + '\n')
                fp.close()

                if show_each_step:
                    train_losses += step_losses
                    train_scores += step_scores
                    step_losses, step_scores = [], []
                else:
                    step_loss = step_losses/validation_frequency
                    step_score = step_scores/validation_frequency
                    train_losses.append(step_loss)
                    train_scores.append(step_score)
                    step_losses, step_scores = 0, 0

            if (i + 1) % num_steps_per_epoch == 0:  # Print and plot results every epoch
                self.train_set.initialize(self.model.session)  # Initialize training iterator every epoch
                if show_each_step:
                    val_freq = validation_frequency
                    start = 0 if pkl_loaded else start_step
                else:
                    val_freq = 1
                    start = 0 if pkl_loaded else start_epoch
                if self.val_set is not None:
                    if verbose:
                        print('[epoch {}/{}]\tTrain loss: {:.6f}  |Train score: {:2.2%}  '
                              '|Eval loss: {:.6f}  |Eval score: {:2.2%}  |LR: {:.6f}  '
                              '|Elapsed time: {:5.0f} sec'
                              .format(self.curr_epoch, self.num_epochs, step_loss, step_score,
                                      eval_loss, eval_score, self.init_learning_rate*self.curr_multiplier,
                                      time.time() - start_time))
                    if len(eval_losses) > 0:
                        plot_learning_curve(train_losses, train_scores,
                                            eval_losses=eval_losses, eval_scores=eval_scores,
                                            name=self.evaluator.name,
                                            loss_threshold=max([2*np.log(self.model.num_classes), min(eval_losses)*2]),
                                            mode=self.evaluator.mode, img_dir=save_dir, annotations=annotations,
                                            start_step=start, validation_frequency=val_freq)

                else:
                    if verbose:
                        print('[epoch {}/{}]\tTrain loss: {:.6f}  |Train score: {:2.2%}  |LR: {:.6f}  '
                              '|Elapsed time: {:5.0f} sec'
                              .format(self.curr_epoch, self.num_epochs, step_loss, step_score,
                                      self.curr_multiplier, time.time() - start_time))
                    plot_learning_curve(step_losses, step_scores, eval_losses=None, eval_scores=None,
                                        name=self.evaluator.name,
                                        loss_threshold=max([self.evaluator.loss_threshold, min(step_losses)*2]),
                                        mode=self.evaluator.mode, img_dir=save_dir, annotations=annotations,
                                        start_step=start, validation_frequency=val_freq)

                self.curr_epoch += 1
                plt.close()

        train_writer.close()
        if verbose:
            print('Total training time: {:.2f} sec'.format(time.time() - start_time))
            print('Best {} {}: {:.4f}'.format('evaluation' if self.val_set is not None
                                              else 'training', self.evaluator.name, self.best_score))

        print('Done.')

        if details:
            train_results['step_losses'] = step_losses
            train_results['step_scores'] = step_scores
            if self.val_set is not None:
                train_results['eval_losses'] = eval_losses
                train_results['eval_scores'] = eval_scores

            return train_results

    def _step(self, handles, merged=None, writer=None, summary=False, log_trace=False):  # Optimization step
        feed_dict = {self.model.is_train: True,
                     self.model.monte_carlo: self.monte_carlo,
                     self.model.augmentation: self.augment_train,
                     self.model.total_steps: self.total_steps,
                     self.learning_rate_multiplier: self.curr_multiplier}
        for h_t, h in zip(self.model.handles, handles):
            feed_dict.update({h_t: h})

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if log_trace else None
        run_metadata = tf.RunMetadata() if log_trace else None

        if summary:  # Write summaries on TensorBoard
            assert merged is not None, 'No merged summary exists.'
            assert writer is not None, 'No summary writer exists.'

            _, loss, Y_true, Y_pred, summaries = self.model.session.run([self.optimization_operation, self.model.loss,
                                                                         self.model.Y_all, self.model.pred, merged],
                                                                        feed_dict=feed_dict,
                                                                        options=run_options,
                                                                        run_metadata=run_metadata)
            writer.add_summary(summaries, self.curr_step + 1)
            writer.flush()
        else:
            _, loss, Y_true, Y_pred, = self.model.session.run([self.optimization_operation, self.model.loss,
                                                               self.model.Y_all, self.model.pred],
                                                              feed_dict=feed_dict,
                                                              options=run_options,
                                                              run_metadata=run_metadata)

        if log_trace:
            assert writer is not None, 'TensorFlow FileWriter must be provided for logging.'
            tracing_dir = os.path.join(writer.get_logdir(), 'tracing')
            if not os.path.exists(tracing_dir):
                os.makedirs(tracing_dir)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
            with open(os.path.join(tracing_dir, 'step_{}.json'.format(self.curr_step + 1)), 'w') as f:
                f.write(chrome_trace)

        return loss, Y_true, Y_pred

    def _update_learning_rate(self):  # Learning rate decay
        warmup_steps = np.around(self.warmup_epoch*self.steps_per_epoch)
        if self.curr_step < warmup_steps:
            self.curr_multiplier = (self.curr_step + 1)/warmup_steps
        else:
            if self.decay_method is not None:
                if self.decay_method.lower() == 'step':  # params: (decay_factor, decay_epoch_0, decay_epoch_1, ...)
                    self.curr_multiplier = 1.0
                    for n in range(len(self.decay_params) - 1):
                        self.curr_multiplier *= np.power(self.decay_params[0],
                                                         np.maximum(np.sign(self.curr_epoch - self.decay_params[n + 1]),
                                                                    0.0))
                elif self.decay_method.lower() == 'exponential':  # params: (decay_factor, decay_every_n_epoch)
                    self.curr_multiplier = self.decay_params[0]**((self.curr_step - warmup_steps)/self.steps_per_epoch
                                                                  / self.decay_params[1])
                elif self.decay_method.lower() == 'poly' or self.decay_method.lower() == 'polynomial':  # param: power
                    power = self.decay_params[0] if isinstance(self.decay_params, (list, tuple)) else self.decay_params
                    total_steps = self.steps_per_epoch*self.num_epochs - warmup_steps
                    self.curr_multiplier = (1 - (self.curr_step - warmup_steps)/total_steps)**power
                else:  # 'cosine', param: num_restarts (annealing)
                    anneal = self.decay_params[0] if isinstance(self.decay_params, (list, tuple)) else self.decay_params
                    anneal = 0 if anneal is None else int(anneal)
                    total_steps = self.steps_per_epoch*self.num_epochs - warmup_steps
                    curr_prog = ((anneal + 1)*(self.curr_step - warmup_steps)/total_steps) % 1.0
                    self.curr_multiplier = 0.5*(1 + np.cos(curr_prog*np.pi))

    def _test_drive(self, save_dir):
        self.train_set.initialize(self.model.session)  # Initialize training iterator
        handles = self.train_set.get_string_handles(self.model.session)  # Get a string handle from training iterator
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        feed_dict = {self.model.is_train: True,
                     self.model.monte_carlo: False,
                     self.model.augmentation: True,
                     self.learning_rate_multiplier: 0.0}
        for h_t, h in zip(self.model.handles, handles):
            feed_dict.update({h_t: h})

        print('Running test epoch...')
        start_time = time.time()
        i = 0
        while True:
            try:
                self.model.session.run([self.optimization_operation, self.model.loss,
                                        self.model.Y_all, self.model.pred],
                                       feed_dict=feed_dict,
                                       options=options,
                                       run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                with open(os.path.join(save_dir, 'logs', 'timeline_{:03}.json'.format(i)), 'w') as f:
                    f.write(chrome_trace)
                i += 1
            except tf.errors.OutOfRangeError:
                break

        print('Test epoch: {:.2f} sec'.format(time.time() - start_time))


class MomentumOptimizer(Optimizer):

    def _optimizer(self, **kwargs):
        momentum = kwargs.get('momentum', 0.9)
        gradient_threshold = kwargs.get('gradient_threshold', None)
        print('Optimizer: SGD with momentum. Initial learning rate: {:.6f}. Gradient threshold: {}'
              .format(self.init_learning_rate, gradient_threshold))

        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum, use_nesterov=True)

        return optimizer


class RMSPropOptimizer(Optimizer):

    def _optimizer(self, **kwargs):
        momentum = kwargs.get('momentum', 0.9)
        decay = 0.9
        eps = 0.001
        gradient_threshold = kwargs.get('gradient_threshold', None)
        print('Optimizer: RMSProp. Initial learning rate: {:.6f}. Gradient threshold: {}'
              .format(self.init_learning_rate, gradient_threshold))

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=decay, momentum=momentum, epsilon=eps)

        return optimizer
