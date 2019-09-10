import os
import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_learning_curve
from tensorflow.python.client import timeline


class Optimizer(object):
    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer.
        :param model: ConvNet, the model to be trained.
        :param train_set: DataSet, training set to be used.
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: DataSet, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra arguments containing training hyperparameters.
            - batch_size: int, batch size for each iteration.
            - num_epochs: int, total number of epochs for training.
            - init_learning_rate: float, initial learning rate.
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
        self.init_learning_rate = kwargs.get('init_learning_rate', 0.01)

        self.warmup_epoch = kwargs.get('learning_warmup_epoch', 0)
        self.decay_method = kwargs.get('learning_rate_decay_method', 'cosine')
        self.decay_params = kwargs.get('learning_rate_decay_params', (0.94, 2))

        self.update_vars = tf.trainable_variables()
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.learning_rate = tf.placeholder(self.model.dtype, name='learning_rate')

        self.optimization_operation = self._optimize_and_update(self._optimizer(**kwargs), **kwargs)

        self._reset()

    def _reset(self):
        self.curr_step = 0
        self.curr_epoch = 1
        self.best_score = self.evaluator.worst_score
        self.learning_rate_update = 0
        self.curr_learning_rate = 0

    @abstractmethod
    def _optimizer(self, **kwargs):
        """
        tf.train.Optimizer.minimize Op for a gradient update
        This should be implemented, and should not be called manually.
        """
        pass

    def _optimize_and_update(self, optimizer, **kwargs):
        gradient_threshold = kwargs.get('gradient_threshold', 5.0)
        loss_scaling_factor = kwargs.get('loss_scaling_factor', 1.0)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.model.num_gpus):
                with tf.device('/gpu:' + str(i)):
                    with tf.name_scope('gpu{}/gradients'.format(i)):
                        loss = loss_scaling_factor*self.model.losses[i]
                        if self.model.dtype is not tf.float32:
                            loss = tf.cast(loss, dtype=self.model.dtype)
                        grads_and_vars = optimizer.compute_gradients(loss, var_list=self.update_vars)
                        grads, gvars = zip(*grads_and_vars)
                        if gradient_threshold is not None:
                            grads, _ = tf.clip_by_global_norm(grads, gradient_threshold*loss_scaling_factor)
                        tower_grads.append([gv for gv in zip(grads, gvars)])
                        tf.get_variable_scope().reuse_variables()

        with tf.device('/cpu:0'):
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
                    grad = grad/loss_scaling_factor

                    # Pointers to the variables are the same for all towers since the variables are shared.
                    avg_vars.append(grads_and_vars[0][1])
                    avg_grads.append(grad)
                # if gradient_threshold is not None:
                #     avg_grads, _ = tf.clip_by_global_norm(avg_grads, gradient_threshold)

                avg_grads_and_vars = [gv for gv in zip(avg_grads, avg_vars)]

        with tf.control_dependencies(self.model.update_ops):
            opt_op = tf.group(*([optimizer.apply_gradients(avg_grads_and_vars)] + self.update_ops))
        return opt_op

    def train(self, save_dir='./tmp', transfer_dir=None, details=False, verbose=True, show_each_step=True, **kwargs):
        if transfer_dir is not None:        # Transfer learning setup
            model_to_load = kwargs.get('model_to_load', None)
            blocks_to_load = kwargs.get('blocks_to_load', None)
            load_logits = kwargs.get('load_logits', False)
            load_moving_average = kwargs.get('load_moving_average', True)
            start_epoch = kwargs.get('start_epoch', 0)

            with tf.Graph().as_default():   # Find variables to be transferred
                tf.train.import_meta_graph(os.path.join(transfer_dir, 'model.ckpt.meta'))
                transfer_variables = tf.global_variables()
                var_names = []
                for var in transfer_variables:
                    if var.name not in var_names:
                        var_names.append(var.name)

            var_list = []
            if blocks_to_load is None:
                for i in range(self.model.num_blocks):
                    var_list += tf.get_collection('block{}_variables'.format(i))
                    var_list += tf.get_collection('block{}_ema_variables'.format(i))
            else:
                for i in blocks_to_load:
                    var_list += tf.get_collection('block{}_variables'.format(i))
                    var_list += tf.get_collection('block{}_ema_variables'.format(i))
            if load_logits:
                var_list += tf.get_collection('block{}_variables'.format(None))
                var_list += tf.get_collection('block{}_ema_variables'.format(None))

            if load_moving_average:
                variables = {}
                for var in var_list:
                    target_emi = var.name.split(':0')[0] + '/ExponentialMovingAverage'
                    if target_emi in var_names:
                        variables[target_emi] = var
                        var_names.remove(target_emi)
                    elif var.name in var_names:
                        variables[var.name.split(':0')[0]] = var
                        var_names.remove(var.name)
            else:
                variables = []
                for var in var_list:
                    if var.name in var_names:
                        variables.append(var)
                        var_names.remove(var.name)

            saver_transfer = tf.train.Saver(variables)

            self.model.session.run(tf.global_variables_initializer())

            if model_to_load is None:       # Find a model to be transferred
                ckpt_to_load = tf.train.latest_checkpoint(transfer_dir)
            elif isinstance(model_to_load, str):
                ckpt_to_load = os.path.join(transfer_dir, model_to_load)
            else:
                fp = open(os.path.join(transfer_dir, 'checkpoints.txt'), 'r')
                ckpt_list = fp.readlines()
                fp.close()
                ckpt_to_load = os.path.join(transfer_dir, ckpt_list[model_to_load].rstrip())

            saver_transfer.restore(self.model.session, ckpt_to_load)
        else:
            start_epoch = 0
            self.model.session.run(tf.global_variables_initializer())

        max_to_keep = kwargs.get('max_to_keep', 5)
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver.export_meta_graph(filename=os.path.join(save_dir, 'model.ckpt.meta'))

        kwargs['monte_carlo'] = False       # Turn off monte carlo dropout for validation

        with tf.device('/cpu:0'):
            with tf.variable_scope('summaries'):    # TensorBoard summaries
                tf.summary.scalar('Loss', self.model.loss)
                tf.summary.scalar('Learning Rate', self.learning_rate)
                tf.summary.scalar('Debug Value', self.model.debug_value)
                tf.summary.image('Input Images',
                                 tf.cast(self.model.input_images*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.image('Augmented Input Images',
                                 tf.cast((tf.clip_by_value(self.model.X_all, -0.5, 0.5) + 0.5)*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.image('Debug Images 0',
                                 tf.cast(self.model.debug_images_0*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.image('Debug Images 1',
                                 tf.cast(self.model.debug_images_1*255, dtype=tf.uint8),
                                 max_outputs=4)
                tf.summary.histogram('Image Histogram', self.model.X[0])
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(save_dir, 'logs'), self.model.session.graph)

        train_results = dict()
        train_size = self.train_set.num_examples
        num_steps_per_epoch = np.ceil(train_size/self.batch_size).astype(int)
        self.steps_per_epoch = num_steps_per_epoch
        num_steps = num_steps_per_epoch*self.num_epochs
        start_step = num_steps_per_epoch*start_epoch

        validation_frequency = kwargs.get('validation_frequency', None)
        summary_frequency = kwargs.get('summary_frequency', None)
        if validation_frequency is None:
            validation_frequency = num_steps_per_epoch
        if summary_frequency is None:
            summary_frequency = num_steps_per_epoch

        num_validations = num_steps//validation_frequency
        last_val_iter = num_validations*validation_frequency

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))
            print('Number of iterations per epoch: {}'.format(num_steps_per_epoch))

        train_losses, train_scores, eval_losses, eval_scores = [], [], [], []
        if show_each_step:
            step_losses, step_scores = [], []
        else:
            step_losses, step_scores = 0, 0
        eval_loss, eval_score = np.inf, 0
        annotations = []

        # self._test_drive(save_dir=save_dir)  # Run test code

        self.train_set.initialize(self.model.session)  # Initialize training iterator
        handles = self.train_set.get_string_handles(self.model.session)  # Get a string handle from training iterator
        tf.get_default_graph().finalize()
        self.curr_epoch += start_epoch
        self.curr_step += start_step
        start_time = time.time()
        for i in range(num_steps - start_step):
            self._update_learning_rate()

            step_loss, step_Y_true, step_Y_pred = self._step(handles, merged=merged, writer=writer,
                                                             summarize=(i + 1) % summary_frequency == 0)
            step_score = self.evaluator.score(step_Y_true, step_Y_pred)
            if show_each_step:
                step_losses.append(step_loss)
                step_scores.append(step_score)
            else:
                step_losses += step_loss
                step_scores += step_score
            self.curr_step += 1

            if (i + 1) % validation_frequency == 0:     # Validation every validation_frequency iterations
                if self.val_set is not None:
                    _, eval_Y_true, eval_Y_pred, eval_loss = self.model.predict(self.val_set, verbose=False,
                                                                                return_images=False, **kwargs)
                    eval_score = self.evaluator.score(eval_Y_true, eval_Y_pred)
                    eval_scores.append(eval_score)
                    eval_losses.append(eval_loss)

                    curr_score = eval_score
                else:
                    curr_score = step_score

                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):  # Save best model
                    self.best_score = curr_score
                    saver.save(self.model.session, os.path.join(save_dir, 'model.ckpt'),
                               global_step=self.curr_step,
                               write_meta_graph=False)

                    if show_each_step:
                        annotations.append((self.curr_step, curr_score))
                    else:
                        annotations.append((self.curr_step//validation_frequency, curr_score))
                    annotations = annotations[-max_to_keep:]
                elif self.curr_step == last_val_iter:  # Save latest model
                    saver.save(self.model.session, os.path.join(save_dir, 'model.ckpt'),
                               global_step=self.curr_step,
                               write_meta_graph=False)

                    if show_each_step:
                        annotations.append((self.curr_step, curr_score))
                    else:
                        annotations.append((self.curr_step // validation_frequency, curr_score))
                    annotations = annotations[-max_to_keep:]

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

            if (i + 1) % num_steps_per_epoch == 0:      # Print and plot results every epoch
                self.train_set.initialize(self.model.session)  # Initialize training iterator every epoch
                # handles = self.train_set.get_string_handles(self.model.session)
                if self.val_set is not None:
                    if verbose:
                        print('[epoch {}/{}]\tTrain loss: {:.6f}  |Train score: {:2.2%}  '
                              '|Eval loss: {:.6f}  |Eval score: {:2.2%}  |LR: {:.6f}  '
                              '|Elapsed time: {:5.0f} sec'
                              .format(self.curr_epoch, self.num_epochs, step_loss, step_score,
                                      eval_loss, eval_score, self.curr_learning_rate,
                                      time.time() - start_time))
                    if len(eval_losses) > 0:
                        if show_each_step:
                            val_freq = validation_frequency
                            start = start_step
                        else:
                            val_freq = 1
                            start = start_epoch
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
                                      self.curr_learning_rate, time.time() - start_time))
                    plot_learning_curve(step_losses, step_scores, eval_losses=None, eval_scores=None,
                                        name=self.evaluator.name,
                                        loss_threshold=max([self.evaluator.loss_threshold, min(step_losses)*2]),
                                        mode=self.evaluator.mode, img_dir=save_dir, annotations=annotations)

                self.curr_epoch += 1
                plt.close()

        if verbose:
            print('Total training time: {:.2f} sec'.format(time.time() - start_time))
            print('Best {} score: {:.4f}'.format('evaluation' if self.val_set is not None
                                                 else 'training', self.best_score))

        ckpt_list = saver.last_checkpoints[::-1]
        fp = open(os.path.join(save_dir, 'checkpoints.txt'), 'w')
        for fname in ckpt_list:
            fp.write(fname.split(os.sep)[-1] + '\n')
        fp.close()

        print('Done.')

        if details:
            train_results['step_losses'] = step_losses
            train_results['step_scores'] = step_scores
            if self.val_set is not None:
                train_results['eval_losses'] = eval_losses
                train_results['eval_scores'] = eval_scores

            return train_results

    def _step(self, handles, merged=None, writer=None, summarize=False):     # Optimization step
        feed_dict = {self.model.is_train: True,
                     self.model.monte_carlo: self.monte_carlo,
                     self.model.augmentation: self.augment_train,
                     self.learning_rate: self.curr_learning_rate}
        for h_t, h in zip(self.model.handles, handles):
            feed_dict.update({h_t: h})

        if summarize:       # Write summaries on TensorBoard
            assert merged is not None, 'No merged summary exists.'
            assert writer is not None, 'No summary writer exists.'
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            _, loss, Y_true, Y_pred, summary = self.model.session.run([self.optimization_operation, self.model.loss,
                                                                       self.model.Y_all, self.model.pred, merged],
                                                                      feed_dict=feed_dict
                                                                      )
            writer.add_summary(summary, self.curr_step + 1)
            # writer.add_run_metadata(run_metadata, 'step{}'.format(self.curr_step + 1))
        else:
            _, loss, Y_true, Y_pred, = self.model.session.run([self.optimization_operation, self.model.loss,
                                                               self.model.Y_all, self.model.pred],
                                                              feed_dict=feed_dict
                                                              )

        return loss, Y_true, Y_pred

    def _test_drive(self, save_dir):
        self.train_set.initialize(self.model.session)  # Initialize training iterator
        handles = self.train_set.get_string_handles(self.model.session)  # Get a string handle from training iterator
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        feed_dict = {self.model.is_train: True,
                     self.model.monte_carlo: False,
                     self.model.augmentation: True,
                     self.learning_rate: 0.0}
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
                                       run_metadata=run_metadata
                                       )
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format(show_memory=False)
                with open(os.path.join(save_dir, 'logs', 'timeline_{:03}.json'.format(i)), 'w') as f:
                    f.write(chrome_trace)
                i += 1
            except tf.errors.OutOfRangeError:
                break

        print('Test epoch: {:.2f} sec'.format(time.time() - start_time))

    def _update_learning_rate(self):  # Learning rate decay
        warmup_steps = np.around(self.warmup_epoch*self.steps_per_epoch)
        if self.curr_step < warmup_steps:
            self.curr_learning_rate = (self.curr_step + 1)/warmup_steps*self.init_learning_rate
        else:
            if self.decay_method is not None:
                if self.decay_method.lower() == 'step':  # params: (decay_epoch_0, decay_epoch_1, ...)
                    if self.curr_epoch >= self.decay_params[self.learning_rate_update]:
                        self.curr_learning_rate *= 0.1
                        self.learning_rate_update += 1
                elif self.decay_method.lower() == 'exponential':  # params: (decay_factor, decay_epoch)
                    if self.curr_epoch//self.decay_params[1] > self.learning_rate_update:
                        self.curr_learning_rate *= self.decay_params[0]
                        self.learning_rate_update += 1
                else:  # 'cosine'
                    total_steps = self.steps_per_epoch*self.num_epochs - warmup_steps
                    self.curr_learning_rate = 0.5*(1 + np.cos((self.curr_step - warmup_steps)*np.pi/total_steps)
                                                   )*self.init_learning_rate


class MomentumOptimizer(Optimizer):

    def _optimizer(self, **kwargs):
        momentum = kwargs.get('momentum', 0.9)
        gradient_threshold = kwargs.get('gradient_threshold', 5.0)
        print('Optimizer: SGD with momentum. Initial learning rate: {}. Gradient threshold: {}'
              .format(self.init_learning_rate, gradient_threshold))

        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum, use_nesterov=True)

        return optimizer
