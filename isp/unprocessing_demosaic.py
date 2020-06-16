"""
Build ISP networks with unprocessing using TensorFlow low-level APIs.
Performs both denoising and demosaicing.
"""

from abc import abstractmethod
import tensorflow.compat.v1 as tf
from isp.unprocessing import Unprocessing
from isp import process, unprocess


class UnprocessingDemosaic(Unprocessing):
    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                with tf.variable_scope('edge'):
                    filters = list()
                    filters.append(tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=tf.float32))
                    filters.append(tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=tf.float32))
                    for i in range(len(filters)):
                        filters[i] = tf.tile(filters[i][..., tf.newaxis, tf.newaxis], [1, 1, 3, 1])
                    self.edge_filters = tf.concat(filters, axis=-1)
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

        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                self.X_all = tf.concat(self.Xs, axis=0, name='x')
                self.Y_all = tf.concat(self.Ys, axis=0, name='y_true')
                self.pred = tf.concat(self.preds, axis=0, name='y_pred')
                self.loss = tf.reduce_mean(self.losses, name='mean_loss')

                self.input_images = tf.concat(self.X_in, axis=0, name='x_in')
                self.input_labels = self.input_images
                self.debug_images.append(self.Y_all)
                self.debug_images.append(self.pred)
                self.debug_images.append(tf.concat(self.Y_edges, axis=0, name='edge_true'))
                self.debug_images.append(tf.concat(self.pred_edges, axis=0, name='edge_pred'))

    def _build_loss(self, **kwargs):
        edge_loss_l1_factor = kwargs.get('edge_loss_l1_factor', 0.0)
        edge_loss_l2_factor = kwargs.get('edge_loss_l2_factor', 0.0)
        denoising_loss_factor = kwargs.get('denoising_loss_factor', 0.0)
        with tf.variable_scope('loss'):
            mask = tf.ones([1, self.input_size[0] - 4, self.input_size[1] - 4, 1])
            mask = tf.pad(mask, [[0, 0], [2, 2], [2, 2], [0, 0]], constant_values=0.0)  # Mask distorted boundaries
            loss = tf.losses.absolute_difference(mask*self.Y, mask*self.pred)
            if edge_loss_l1_factor > 0.0 or edge_loss_l2_factor > 0.0:
                with tf.variable_scope('edge_loss'):
                    tr = kwargs.get('edge_loss_true_ratio', 0.0)
                    edge_y = tf.nn.depthwise_conv2d(self.Y, self.edge_filters,
                                                    strides=[1, 1, 1, 1], padding='SAME')
                    edge_pred = tf.nn.depthwise_conv2d(self.pred, self.edge_filters,
                                                       strides=[1, 1, 1, 1], padding='SAME')
                    edge_y *= mask
                    edge_pred *= mask

                    num_filters = self.edge_filters.get_shape().as_list()[-1]
                    y_sum = 0
                    pred_sum = 0
                    for i in range(num_filters):
                        y_sum += edge_y[..., i*3:(i + 1)*3]**2
                        pred_sum += edge_pred[..., i*3:(i + 1)*3]**2
                    self.Y_edges.append(tf.math.sqrt(y_sum))
                    self.pred_edges.append(tf.math.sqrt(pred_sum))

                    true_edge = tf.math.sqrt(tf.math.reduce_sum(edge_y**2, axis=-1, keepdims=True))
                    edge_l1 = tf.math.reduce_mean(((1.0 - tr) + tr*true_edge)*tf.math.abs(edge_y - edge_pred))
                    edge_l2 = tf.math.reduce_mean(((1.0 - tr) + tr*true_edge)*tf.math.pow(edge_y - edge_pred, 2))
                    edge_l2 = tf.math.sqrt(edge_l2 + 1e-5)
                    loss += edge_l1*edge_loss_l1_factor + edge_l2*edge_loss_l2_factor
            if denoising_loss_factor > 0.0 and self.denoised is not None:
                loss += tf.losses.absolute_difference(mask*self.Y_mosaic, mask*self.denoised,
                                                      weights=denoising_loss_factor)

            l1_factor = kwargs.get('l1_reg', 0e-8)
            l2_factor = kwargs.get('l2_reg', 1e-4)
            variables = tf.get_collection('weight_variables')
            if kwargs.get('bias_norm_decay', False):
                variables += tf.get_collection('bias_variables') + tf.get_collection('norm_variables')
            with tf.variable_scope('l1_loss'):
                if l1_factor > 0.0:
                    l1_factor = tf.constant(l1_factor, dtype=tf.float32, name='L1_factor')
                    l1_reg_loss = l1_factor * tf.accumulate_n([tf.reduce_sum(tf.math.abs(var)) for var in variables])
                else:
                    l1_reg_loss = tf.constant(0.0, dtype=tf.float32, name='0')
            with tf.variable_scope('l2_loss'):
                if l2_factor > 0.0:
                    l2_factor = tf.constant(l2_factor, dtype=tf.float32, name='L2_factor')
                    l2_reg_loss = l2_factor * tf.math.accumulate_n([tf.nn.l2_loss(var) for var in variables])
                else:
                    l2_reg_loss = tf.constant(0.0, dtype=tf.float32, name='0')
            return loss + l1_reg_loss + l2_reg_loss

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

    def unprocess(self, image):
        with tf.name_scope(None, 'unprocess'):
            image.shape.assert_is_compatible_with([None, None, 3])

            # Randomly creates image metadata.
            rgb2cam = unprocess.random_ccm()
            cam2rgb = tf.matrix_inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = unprocess.random_gains()

            # Approximately inverts global tone mapping.
            image = unprocess.inverse_smoothstep(image)
            # Inverts gamma compression.
            image = unprocess.gamma_expansion(image)
            # Inverts color correction.
            image = unprocess.apply_ccm(image, rgb2cam)
            # Approximately inverts white balance and brightening.
            image = unprocess.safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
            # Clips saturated pixels.
            image = tf.clip_by_value(image, 0.0, 1.0)
            # Applies a Bayer mosaic.
            bayer_image = unprocess.mosaic(image)

            metadata = {
                'cam2rgb': cam2rgb,
                'rgb_gain': rgb_gain,
                'red_gain': red_gain,
                'blue_gain': blue_gain,
            }
            return image, bayer_image, metadata

    def unprocess_images(self, image):
        image, bayer_image, metadata = self.unprocess(image)
        shot_noise, read_noise = unprocess.random_noise_levels()
        noisy_img = unprocess.add_noise(bayer_image, shot_noise, read_noise)
        variance = shot_noise*noisy_img + read_noise

        metadata = [v for v in metadata.values()]
        return image, bayer_image, noisy_img, variance, metadata

