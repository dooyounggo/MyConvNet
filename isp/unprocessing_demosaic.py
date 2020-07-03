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
        dtypes = (tf.float32, tf.float32)
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        self._init_unprocessing(**kwargs)
        self.denoising_losses = []
        self._make_filters()
        self._set_next_elements(dtypes, output_shapes)
        self._init_vgg_net(**kwargs)
        vgg_input_gt = dict()
        vgg_input_pred = dict()
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                self._curr_device = i
                self._curr_block = None
                self._curr_dependent_op = 0  # For ops with dependencies between GPUs such as BN
                device = '/{}:'.format(self.compute_device) + str(i)
                with tf.device(device):
                    with tf.name_scope(self.compute_device + '_' + str(i) + '/'):
                        self.X, _ = self.next_elements[device]
                        self.X_in.append(self.X)

                        self.X = self.zero_pad(self.X, pad_value=self.pad_value)
                        self.X = tf.cond(self.augmentation,
                                         lambda: self.augment_images(self.X, **kwargs),
                                         lambda: self.center_crop(self.X),
                                         name='augmentation')

                        image, bayer_img, noisy_img, variance, metadata = tf.map_fn(self.unprocess_images, self.X,
                                                                                    dtype=(tf.float32, tf.float32,
                                                                                           tf.float32, tf.float32,
                                                                                           [tf.float32, tf.float32,
                                                                                            tf.float32, tf.float32,
                                                                                            tf.float32, tf.float32]),
                                                                                    parallel_iterations=32,
                                                                                    back_prop=False)

                        self.Y = self.process(image, metadata[2], metadata[3], metadata[0],
                                              simple=self.simple_unprocessing)
                        self.Y.set_shape([None] + list(self.input_size))
                        vgg_input_gt[device] = self.Y

                        self.Y_mosaic = process.process(bayer_img, metadata[2], metadata[3], metadata[0],
                                                        simple=self.simple_unprocessing)
                        self.Y_mosaic.set_shape([None] + list(self.input_size))
                        noisy = process.process(noisy_img, metadata[2], metadata[3], metadata[0],
                                                simple=self.simple_unprocessing)
                        noisy.set_shape([None] + list(self.input_size))
                        self.Xs.append(noisy)
                        self.Ys.append(self.Y)

                        self.X = tf.concat([noisy_img, variance], axis=-1)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + '_' + str(i))):
                                self.X = tf.cast(self.X, dtype=self.dtype)

                        self._shot_noise_tensor = metadata[4]
                        self._read_noise_tensor = metadata[5]
                        with tf.name_scope('nn'):
                            self.d = self._build_model()
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + '_' + str(i))):
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        if self.channel_first:
                            self.d['pred'] = tf.transpose(self.d['pred'], perm=[0, 2, 3, 1])
                        tf.get_variable_scope().reuse_variables()

                        self.dicts.append(self.d)
                        self.pred = self.process(self.d['pred'], metadata[2], metadata[3], metadata[0],
                                                 simple=self.simple_unprocessing)
                        self.pred.set_shape([None] + list(self.input_size))
                        if 'denoised' in self.d:
                            if self.dtype is not tf.float32:
                                with tf.name_scope('{}/cast/'.format(self.compute_device + '_' + str(i))):
                                    self.d['denoised'] = tf.cast(self.d['denoised'], dtype=tf.float32)
                            if self.channel_first:
                                self.d['denoised'] = tf.transpose(self.d['denoised'], perm=[0, 2, 3, 1])
                            self.denoised = process.process(self.d['denoised'], metadata[2], metadata[3], metadata[0],
                                                            simple=self.simple_unprocessing)
                            self.denoised.set_shape([None] + list(self.input_size))
                        else:
                            self.denoised = None
                        self.preds.append(self.pred)
                        vgg_input_pred[device] = self.pred

                        self.losses.append(self._build_loss(**kwargs))

        self._build_perceptual_loss(vgg_input_gt, vgg_input_pred, **kwargs)
        self._make_debug_images()
        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                self.debug_values.append(tf.reduce_mean(self.denoising_losses, name='denoising_loss'))

    def _build_loss(self, **kwargs):
        with tf.variable_scope('loss'):
            denoising_loss_factor = kwargs.get('denoising_loss_factor', 0.0)
            loss = self._loss_fn(**kwargs)
            if denoising_loss_factor > 0.0 and self.denoised is not None:
                deno_loss = tf.losses.absolute_difference(self.Y_mosaic, self.denoised, weights=denoising_loss_factor)
                loss += deno_loss
                self.denoising_losses.append(deno_loss)
            return loss

    @abstractmethod
    def _build_model(self):
        """
        Build model.
        This should be implemented.
        :return dict containing tensors. Must include 'pred' tensor.
        """
        pass

    def process(self, images, red_gains, blue_gains, cam2rgbs, simple=False):
        images.shape.assert_is_compatible_with([None, None, None, 3])
        with tf.name_scope(None, 'process'):
            if not simple:
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
