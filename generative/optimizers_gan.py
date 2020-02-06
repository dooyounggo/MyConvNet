"""
Optimizer class for GANs.
"""

from optimizers import *


class GANOptimizer(Optimizer):
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
        generator_scaling_factor = kwargs.get('generator_scaling_factor', 1.0)
        weight_decay = kwargs.get('base_weight_decay', 0.0)*self.batch_size/256
        weight_decay_scheduling = kwargs.get('weight_decay_scheduling', True)

        vars_d = []
        vars_g = []
        for i in range(self.model.num_blocks - self.model.num_blocks_g):
            vars_d += tf.get_collection('block_{}_variables'.format(i))
        vars_d += tf.get_collection('block_{}_variables'.format(None))
        for i in range(self.model.num_blocks - self.model.num_blocks_g, self.model.num_blocks):
            vars_g += tf.get_collection('block_{}_variables'.format(i))

        update_vars_d = []
        update_vars_g = []
        for var in vars_d:
            if var.trainable:
                update_vars_d.append(var)
        for var in vars_g:
            if var.trainable:
                update_vars_g.append(var)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.model.gpu_offset, self.model.num_gpus + self.model.gpu_offset):
                with tf.device('/gpu:' + str(i)):
                    with tf.variable_scope('gpu{}/gradients'.format(i)):
                        loss_d = self.model.losses[i - self.model.gpu_offset]
                        loss_g = self.model.losses_g[i - self.model.gpu_offset]
                        if loss_scaling_factor > 1.0:
                            loss_d *= loss_scaling_factor
                            loss_g *= loss_scaling_factor
                        if generator_scaling_factor != 1.0:
                            loss_g *= generator_scaling_factor
                        if self.model.dtype is not tf.float32:
                            loss_d = tf.cast(loss_d, dtype=self.model.dtype)
                            loss_g = tf.cast(loss_g, dtype=self.model.dtype)
                        grads_and_vars_d = optimizer.compute_gradients(loss_d, var_list=update_vars_d)
                        grads_and_vars_g = optimizer.compute_gradients(loss_g, var_list=update_vars_g)
                        grads_and_vars = grads_and_vars_d + grads_and_vars_g
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

        if weight_decay > 0.0:
            variables = tf.get_collection('weight_variables')
            if kwargs.get('bias_norm_decay', False):
                variables += tf.get_collection('bias_variables') + tf.get_collection('norm_variables')
            with tf.variable_scope('weight_decay'):
                weight_decay = tf.constant(weight_decay, dtype=tf.float32, name='weight_decay_factor')
                if weight_decay_scheduling:
                    weight_decay = self.learning_rate_multiplier*weight_decay
                with tf.control_dependencies(self.model.update_ops + self.update_ops):
                    with tf.control_dependencies([optimizer.apply_gradients(avg_grads_and_vars,
                                                                            global_step=self.model.global_step)]):
                        decay_ops = []
                        for var in variables:
                            if var.trainable:
                                decay_op = var.assign_sub(weight_decay*var)
                                decay_ops.append(decay_op)
                        opt_op = tf.group(decay_ops)
        else:
            with tf.control_dependencies(self.model.update_ops + self.update_ops):
                opt_op = optimizer.apply_gradients(avg_grads_and_vars, global_step=self.model.global_step)
        return opt_op

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

            _, loss, loss_g, pred, summaries = self.model.session.run([self.optimization_operation, self.model.loss,
                                                                       self.model.loss_g, self.model.pred, merged],
                                                                      feed_dict=feed_dict,
                                                                      options=run_options,
                                                                      run_metadata=run_metadata)
            writer.add_summary(summaries, self.curr_step + 1)
            writer.flush()
        else:
            _, loss, loss_g, pred, = self.model.session.run([self.optimization_operation, self.model.loss,
                                                             self.model.loss_g, self.model.pred],
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

        return loss, loss_g, pred


class MomentumOptimizer(GANOptimizer):
    def _optimizer(self, **kwargs):
        momentum = kwargs.get('momentum', 0.9)
        gradient_threshold = kwargs.get('gradient_threshold', None)
        print('Optimizer: SGD with momentum. Initial learning rate: {:.6f}. Gradient threshold: {}'
              .format(self.init_learning_rate, gradient_threshold))

        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum, use_nesterov=True)

        return optimizer


class RMSPropOptimizer(GANOptimizer):
    def _optimizer(self, **kwargs):
        momentum = kwargs.get('momentum', 0.9)
        decay = 0.9
        eps = 0.001
        gradient_threshold = kwargs.get('gradient_threshold', None)
        print('Optimizer: RMSProp. Initial learning rate: {:.6f}. Gradient threshold: {}'
              .format(self.init_learning_rate, gradient_threshold))

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=decay, momentum=momentum, epsilon=eps)

        return optimizer


class AdamOptimizer(GANOptimizer):
    def _optimizer(self, **kwargs):
        momentum = kwargs.get('momentum', 0.9)
        decay = 0.999
        eps = 0.001
        gradient_threshold = kwargs.get('gradient_threshold', None)
        print('Optimizer: Adam. Initial learning rate: {:.6f}. Gradient threshold: {}'
              .format(self.init_learning_rate, gradient_threshold))

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=momentum, beta2=decay, epsilon=eps)

        return optimizer
