"""
Build ISP networks with unprocessing using TensorFlow low-level APIs.
Reference: https://github.com/google-research/google-research/tree/master/unprocessing
"""

from abc import abstractmethod
import tensorflow.compat.v1 as tf
import numpy as np
from convnet import ConvNet
from isp import process, unprocess


class Unprocessing(ConvNet):
    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                self._curr_device = i
                self._curr_block = 0
                self._num_blocks = 1  # Total number of blocks
                self._curr_dependent_op = 0  # For ops with dependencies between GPUs such as BN
                with tf.device('/{}:'.format(self.compute_device) + str(i)):
                    with tf.name_scope('{}'.format(self.compute_device + str(i))):
                        handle = tf.placeholder(tf.string, shape=[], name='handle')  # Handle for the feedable iterator
                        self.handles.append(handle)
                        iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.float32),
                                                                       output_shapes=output_shapes)
                        self.X, _ = iterator.get_next()
                        self.X_in.append(self.X)

                        if self._padded_size[0] > self.input_size[0] or self._padded_size[1] > self.input_size[1]:
                            self.X = self.zero_pad(self.X, pad_value=self.pad_value)
                        self.X = tf.cond(self.augmentation,
                                         lambda: self.augment_images(self.X, **kwargs),
                                         lambda: self.center_crop(self.X),
                                         name='augmentation')

                        bayer_img, noisy_img, variance, metadata = tf.map_fn(self.unprocess_images, self.X,
                                                                             dtype=(tf.float32, tf.float32, tf.float32,
                                                                                    [tf.float32, tf.float32,
                                                                                     tf.float32, tf.float32]),
                                                                             parallel_iterations=32, back_prop=False)

                        self.Y = process.process(bayer_img, metadata[2], metadata[3], metadata[0])
                        self.Y.set_shape([None] + list(self.input_size))
                        noisy = process.process(noisy_img, metadata[2], metadata[3], metadata[0])
                        noisy.set_shape([None] + list(self.input_size))
                        self.Xs.append(noisy)
                        self.Ys.append(self.Y)

                        self.X = tf.concat([noisy_img, variance], axis=-1)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + str(i))):
                                self.X = tf.cast(self.X, dtype=self.dtype)

                        with tf.name_scope('nn'):
                            self.d = self._build_model()
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + str(i))):
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        if self.channel_first:
                            self.d['pred'] = tf.transpose(self.d['pred'], perm=[0, 2, 3, 1])
                        tf.get_variable_scope().reuse_variables()

                        self.dicts.append(self.d)
                        self.pred = process.process(self.d['pred'], metadata[2], metadata[3], metadata[0])
                        self.pred.set_shape([None] + list(self.input_size))
                        self.preds.append(self.pred)
                        self.losses.append(self._build_loss(**kwargs))

        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                self.X_all = tf.concat(self.Xs, axis=0, name='x')
                self.Y_all = tf.concat(self.Ys, axis=0, name='y_true')
                self.pred = tf.concat(self.preds, axis=0, name='y_pred')
                self.loss = tf.reduce_mean(self.losses, name='mean_loss')

                self.input_images = tf.concat(self.X_in, axis=0, name='x_in')
                self.input_labels = self.input_images
                self.debug_images_0 = self.Y_all
                self.debug_images_1 = self.pred

    def _build_loss(self, **kwargs):
        with tf.variable_scope('loss'):
            loss = tf.losses.absolute_difference(self.Y, self.pred)
        return loss

    @abstractmethod
    def _build_model(self):
        """
        Build model.
        This should be implemented.
        :return dict containing tensors. Must include 'pred' tensor.
        """
        pass

    def augment_images(self, x, **kwargs):
        rand_blur = kwargs.get('rand_blur_stddev', 0.0) > 0.0
        rand_affine = kwargs.get('rand_affine', False)
        rand_crop = kwargs.get('rand_crop', False)

        if rand_blur:
            x = self.gaussian_blur(x, **kwargs)

        if rand_affine:
            x, mask = self.affine_augment(x, None, **kwargs)

        if rand_crop:
            x, mask = self.rand_crop(x, None, **kwargs)
        else:
            x = self.center_crop(x)

        return x

    def unprocess_images(self, image):
        image, metadata = unprocess.unprocess(image)
        shot_noise, read_noise = unprocess.random_noise_levels()
        noisy_img = unprocess.add_noise(image, shot_noise, read_noise)
        variance = shot_noise*noisy_img + read_noise

        metadata = [v for v in metadata.values()]
        return image, noisy_img, variance, metadata

