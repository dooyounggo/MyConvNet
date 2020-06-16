"""
Build GAN using TensorFlow low-level APIs.
"""

import time
from abc import abstractmethod
import tensorflow.compat.v1 as tf
import numpy as np
from convnet import ConvNet


class GAN(ConvNet):
    @property
    def num_blocks_g(self):
        return self._num_blocks_g

    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        self.losses_g = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                self._curr_device = i
                self._curr_block = None
                self._num_blocks_g = 1  # Number of generator blocks
                self._curr_dependent_op = 0  # For ops with dependencies between GPUs such as BN
                with tf.device('/{}:'.format(self.compute_device) + str(i)):
                    with tf.name_scope('{}'.format(self.compute_device + str(i))):
                        handle = tf.placeholder(tf.string, shape=[], name='handle')  # Handle for the feedable iterator
                        self.handles.append(handle)
                        iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.float32),
                                                                       output_shapes=output_shapes)
                        self.X, self.Y = iterator.get_next()  # Y is latent vectors

                        # FIXME: Fake label generation
                        self._broadcast_shape = [self.num_classes]  # num_classes is the size of the latent vector.
                        self.Y = tf.map_fn(self._broadcast_nans, self.Y)  # Broadcasting for NaNs
                        self.Y = tf.where(tf.is_nan(self.Y), tf.random_uniform(tf.shape(self.Y), -1.0, 1.0), self.Y)

                        self.X_in.append(self.X)
                        self.Y_in.append(self.Y)

                        if self._padded_size[0] > self.input_size[0] or self._padded_size[1] > self.input_size[1]:
                            self.X = self.zero_pad(self.X, pad_value=self.pad_value)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        self.X = tf.cond(self.augmentation,
                                         lambda: self.augment_images(self.X, **kwargs),
                                         lambda: self.center_crop(self.X),
                                         name='augmentation')

                        self.Xs.append(self.X)
                        self.Ys.append(self.Y)

                        self.X *= self.scale_factor  # Scale images
                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + str(i))):
                                self.X = tf.cast(self.X, dtype=self.dtype)
                                self.Y = tf.cast(self.Y, dtype=self.dtype)

                        with tf.name_scope('nn'):
                            d_real = self._build_model()
                            self.d = self._build_model_g()
                            self.X = self.d['generate']
                            tf.get_variable_scope().reuse_variables()
                            d_fake = self._build_model()
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + str(i))):
                                d_real['logits'] = tf.cast(d_real['logits'], dtype=tf.float32)
                                d_fake['logits'] = tf.cast(d_fake['logits'], dtype=tf.float32)
                                self.d['generate'] = tf.cast(self.d['generate'], dtype=tf.float32)

                        self.d.update(d_fake)
                        self.dicts.append(self.d)
                        self.logits_real = d_real['logits']
                        self.logits_fake = d_fake['logits']
                        self.pred = self.d['generate']
                        losses = self._build_loss(**kwargs)
                        self.losses.append(losses[0])
                        self.losses_g.append(losses[1])
                        self.preds.append(self.pred)

                        # self.bytes_in_use.append(tf_contrib.memory_stats.BytesInUse())

        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                self.X_all = tf.concat(self.Xs, axis=0, name='x') + self.image_mean
                self.Y_all = tf.concat(self.Ys, axis=0, name='y_true')
                self.valid_mask = tf.ones_like(self.Y_all, dtype=tf.float32)
                self.pred = tf.concat(self.preds, axis=0, name='y_pred')/2 + 0.5
                self.loss = tf.reduce_mean(self.losses, name='mean_loss')
                self.loss_g = tf.reduce_mean(self.losses_g, name='mean_loss_g')

                self.input_images = tf.concat(self.X_in, axis=0, name='x_in')
                self.input_labels = tf.concat(self.Y_in, axis=0, name='y_in')
                self.debug_images.append(self.pred)

    @abstractmethod
    def _build_model(self):
        """
        Build model of discriminator networks.
        This should be implemented.
        :return dict containing tensors. Must include 'logits' and 'pred' tensors.
        """
        pass

    @abstractmethod
    def _build_model_g(self):
        """
        Build model of generator networks.
        This should be implemented.
        :return dict containing tensors. Must include 'generate' tensor.
        """
        pass

    def _build_loss(self, **kwargs):
        ls_factor = kwargs.get('label_smoothing', 0.0)

        w = self.loss_weights
        if w is None:
            w = np.ones(2, dtype=np.float32)
        else:
            w = np.array(w, dtype=np.float32)
        print('\nLoss weights: ', w)

        with tf.variable_scope('loss'):
            w = tf.constant(w, dtype=tf.float32, name='class_weights')

            labels_real = tf.ones_like(self.logits_real, dtype=tf.float32)
            labels_fake = tf.zeros_like(self.logits_fake, dtype=tf.float32)
            if ls_factor > 0.0:
                labels_real = self._label_smoothing(labels_real, ls_factor)
                labels_fake = self._label_smoothing(labels_fake, ls_factor)

            losses_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=self.logits_real)
            losses_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=self.logits_fake)
            losses_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=self.logits_fake)
            loss_d = tf.reduce_mean(w[1]*losses_real + w[0]*losses_fake)
            loss_g = tf.reduce_mean(w[0]*losses_g)

        return loss_d, loss_g

    def _broadcast_nans(self, y):
        return tf.broadcast_to(y, self._broadcast_shape)

    def _label_smoothing(self, labels, ls_factor, name='label_smoothing'):
        with tf.variable_scope(name):
            ls_factor = tf.constant(ls_factor, dtype=tf.float32, name='label_smoothing_factor')
            labels = labels*(1.0 - ls_factor)  # One-sided smoothing
        return labels

    def predict(self, dataset, verbose=False, return_images=True, **kwargs):
        batch_size = dataset.batch_size
        augment_test = kwargs.get('augment_test', False)

        pred_size = dataset.num_examples
        num_steps = np.ceil(pred_size/batch_size).astype(int)
        monte_carlo = kwargs.get('monte_carlo', False)

        dataset.initialize(self.session)
        handles = dataset.get_string_handles(self.session)

        if verbose:
            print('Running prediction loop...')

        feed_dict = {self.is_train: False,
                     self.monte_carlo: monte_carlo,
                     self.augmentation: augment_test,
                     self.total_steps: num_steps}
        for h_t, h in zip(self.handles, handles):
            feed_dict.update({h_t: h})

        if return_images:
            _X = np.zeros([pred_size] + list(self.input_size), dtype=np.float32)
        else:
            _X = np.zeros([pred_size] + [4, 4, 3], dtype=np.float32)  # Dummy images
        _loss_g = np.zeros(pred_size, dtype=np.float32)
        _pred = np.zeros([pred_size] + self.pred.get_shape().as_list()[1:], dtype=np.float32)
        _loss_pred = np.zeros(pred_size, dtype=np.float32)
        start_time = time.time()
        for i in range(num_steps):
            try:
                X, loss_g, pred, loss_pred = self.session.run([self.X_all, self.loss_g, self.pred, self.loss],
                                                              feed_dict=feed_dict)
                sidx = i*batch_size
                eidx = (i + 1)*batch_size
                if return_images:
                    _X[sidx:eidx] = X
                _loss_g[sidx:eidx] = loss_g
                _pred[sidx:eidx] = pred
                _loss_pred[sidx:eidx] = loss_pred
            except tf.errors.OutOfRangeError:
                if verbose:
                    print('The last iteration ({} data) has been ignored'.format(pred_size - i*batch_size))

        if verbose:
            print('Total prediction time: {:.2f} sec'.format(time.time() - start_time))

        _loss_pred = np.mean(_loss_pred, axis=0)

        return _X, _loss_g, _pred, _loss_pred
