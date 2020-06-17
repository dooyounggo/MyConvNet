"""
Build ISP networks with unprocessing using TensorFlow low-level APIs.
Performs both denoising and demosaicing.
"""

from abc import abstractmethod
import tensorflow.compat.v1 as tf
from isp.unprocessing import Unprocessing
from isp import process


class UnprocessingDemosaic(Unprocessing):
    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        self._make_filters()
        self.Y_edges = []
        self.pred_edges = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                self._curr_device = i
                self._curr_block = None
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

                        image, bayer_img, noisy_img, variance, metadata = tf.map_fn(self.unprocess_images, self.X,
                                                                                    dtype=(tf.float32, tf.float32,
                                                                                           tf.float32, tf.float32,
                                                                                           [tf.float32, tf.float32,
                                                                                            tf.float32, tf.float32]),
                                                                                    parallel_iterations=32,
                                                                                    back_prop=False)

                        self.Y = self.process(image, metadata[2], metadata[3], metadata[0])
                        self.Y.set_shape([None] + list(self.input_size))
                        self.Y_mosaic = process.process(bayer_img, metadata[2], metadata[3], metadata[0])
                        self.Y_mosaic.set_shape([None] + list(self.input_size))
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
                        self.pred = self.process(self.d['pred'], metadata[2], metadata[3], metadata[0])
                        self.pred.set_shape([None] + list(self.input_size))
                        if 'denoised' in self.d:
                            if self.dtype is not tf.float32:
                                with tf.name_scope('{}/cast/'.format(self.compute_device + str(i))):
                                    self.d['denoised'] = tf.cast(self.d['denoised'], dtype=tf.float32)
                            if self.channel_first:
                                self.d['denoised'] = tf.transpose(self.d['denoised'], perm=[0, 2, 3, 1])
                            self.denoised = process.process(self.d['denoised'], metadata[2], metadata[3], metadata[0])
                            self.denoised.set_shape([None] + list(self.input_size))
                        else:
                            self.denoised = None
                        self.preds.append(self.pred)
                        self.losses.append(self._build_loss(**kwargs))

        self._make_debug_images()

    def _build_loss(self, **kwargs):
        with tf.variable_scope('loss'):
            denoising_loss_factor = kwargs.get('denoising_loss_factor', 0.0)
            loss = self._loss_fn(**kwargs)
            if denoising_loss_factor > 0.0 and self.denoised is not None:
                loss += tf.losses.absolute_difference(self.Y_mosaic, self.denoised, weights=denoising_loss_factor)
            return loss

    @abstractmethod
    def _build_model(self):
        """
        Build model.
        This should be implemented.
        :return dict containing tensors. Must include 'pred' tensor.
        """
        pass

    def process(self, images, red_gains, blue_gains, cam2rgbs):
        images.shape.assert_is_compatible_with([None, None, None, 3])
        with tf.name_scope(None, 'process'):
            # White balance.
            images = self.apply_gains(images, red_gains, blue_gains)
            images = tf.clip_by_value(images, 0.0, 1.0)
            # Color correction.
            images = process.apply_ccms(images, cam2rgbs)
            # Gamma compression.
            images = tf.clip_by_value(images, 0.0, 1.0)
            images = process.gamma_compression(images)
        return images

    def apply_gains(self, images, red_gains, blue_gains):
        green_gains = tf.ones_like(red_gains)
        gains = tf.stack([red_gains, green_gains, blue_gains], axis=-1)
        gains = gains[:, tf.newaxis, tf.newaxis, :]
        return images*gains
