"""
Build ISP networks with unprocessing using TensorFlow low-level APIs.
Reference: https://github.com/google-research/google-research/tree/master/unprocessing
"""

from abc import abstractmethod
import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
from convnet import ConvNet
from isp import process, unprocess
from subsets.subset_functions import to_int


class Unprocessing(ConvNet):
    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        self._init_unprocessing(**kwargs)
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

                        _, bayer_img, noisy_img, variance, metadata = tf.map_fn(self.unprocess_images, self.X,
                                                                                dtype=(tf.float32, tf.float32,
                                                                                       tf.float32, tf.float32,
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

        self._make_debug_images()

    def _build_loss(self, **kwargs):
        with tf.variable_scope('loss'):
            loss = self._loss_fn(**kwargs)
        return loss

    def _loss_fn(self, **kwargs):
        edge_loss_l1_factor = kwargs.get('edge_loss_l1_factor', 0.0)
        edge_loss_l2_factor = kwargs.get('edge_loss_l2_factor', 0.0)
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

        l1_factor = kwargs.get('l1_reg', 0e-8)
        l2_factor = kwargs.get('l2_reg', 1e-4)
        variables = tf.get_collection('weight_variables')
        if kwargs.get('bias_norm_decay', False):
            variables += tf.get_collection('bias_variables') + tf.get_collection('norm_variables')
        with tf.variable_scope('l1_loss'):
            if l1_factor > 0.0:
                l1_factor = tf.constant(l1_factor, dtype=tf.float32, name='L1_factor')
                l1_reg_loss = l1_factor*tf.accumulate_n([tf.reduce_sum(tf.math.abs(var)) for var in variables])
            else:
                l1_reg_loss = tf.constant(0.0, dtype=tf.float32, name='0')
        with tf.variable_scope('l2_loss'):
            if l2_factor > 0.0:
                l2_factor = tf.constant(l2_factor, dtype=tf.float32, name='L2_factor')
                l2_reg_loss = l2_factor*tf.math.accumulate_n([tf.nn.l2_loss(var) for var in variables])
            else:
                l2_reg_loss = tf.constant(0.0, dtype=tf.float32, name='0')
        return loss + l1_reg_loss + l2_reg_loss

    def _init_unprocessing(self, **kwargs):
        ccm = kwargs.get('color_correction_matrix', None)
        rgb_gain = kwargs.get('rgb_gain', None)
        red_gain = kwargs.get('red_gain', None)
        blue_gain = kwargs.get('blue_gain', None)
        shot_noise = kwargs.get('shot_noise', None)
        read_noise = kwargs.get('read_noise', None)
        if ccm is None:
            ccm = [[np.nan, np.nan, np.nan],
                   [np.nan, np.nan, np.nan],
                   [np.nan, np.nan, np.nan]]
        if rgb_gain is None:
            rgb_gain = np.nan
        if red_gain is None:
            red_gain = np.nan
        if blue_gain is None:
            blue_gain = np.nan
        if shot_noise is None:
            shot_noise = np.nan
        if read_noise is None:
            read_noise = np.nan
        self._ccm = ccm
        self._rgb_gain = rgb_gain
        self._red_gain = red_gain
        self._blue_gain = blue_gain
        self._shot_noise = shot_noise
        self._read_noise = read_noise

    def _make_filters(self):
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

    def _make_debug_images(self):
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

                edge_true = tf.concat(self.Y_edges, axis=0, name='edge_true')
                edge_pred = tf.concat(self.pred_edges, axis=0, name='edge_pred')
                self.debug_images.append(edge_true)
                self.debug_images.append(edge_pred)
                self.debug_images.append(tf.math.abs(self.Y_all - self.pred, name='image_diff'))
                self.debug_images.append(tf.math.abs(edge_true - edge_pred, name='edge_diff'))

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

    def unprocess(self, image):
        with tf.name_scope(None, 'unprocess'):
            image.shape.assert_is_compatible_with([None, None, 3])

            # Randomly creates image metadata.
            rgb2cam = unprocess.random_ccm()
            rgb_gain, red_gain, blue_gain = unprocess.random_gains()

            rgb2cam = tf.cond(self.is_train,
                              true_fn=lambda: rgb2cam,
                              false_fn=lambda: tf.where(tf.math.is_nan(self._ccm), rgb2cam, self._ccm))
            rgb_gain = tf.cond(self.is_train,
                               true_fn=lambda: rgb_gain,
                               false_fn=lambda: tf.where(tf.math.is_nan(self._rgb_gain), rgb_gain, self._rgb_gain))
            red_gain = tf.cond(self.is_train,
                               true_fn=lambda: red_gain,
                               false_fn=lambda: tf.where(tf.math.is_nan(self._red_gain), red_gain, self._red_gain))
            blue_gain = tf.cond(self.is_train,
                                true_fn=lambda: blue_gain,
                                false_fn=lambda: tf.where(tf.math.is_nan(self._blue_gain), blue_gain, self._blue_gain))

            cam2rgb = tf.matrix_inverse(rgb2cam)

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
        shot_noise = tf.cond(self.is_train,
                             true_fn=lambda: shot_noise,
                             false_fn=lambda: tf.where(tf.math.is_nan(self._shot_noise), shot_noise, self._shot_noise))
        read_noise = tf.cond(self.is_train,
                             true_fn=lambda: read_noise,
                             false_fn=lambda: tf.where(tf.math.is_nan(self._read_noise), read_noise, self._read_noise))
        noisy_img = unprocess.add_noise(bayer_image, shot_noise, read_noise)
        variance = shot_noise*noisy_img + read_noise

        metadata = [v for v in metadata.values()]
        return image, bayer_image, noisy_img, variance, metadata

    def save_results(self, dataset, save_dir, epoch, max_examples=None, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        if max_examples is None:
            num_examples = min(8, dataset.num_examples)
        else:
            num_examples = min(max_examples, dataset.num_examples)

        noisy, gt, pred, _ = self.predict(dataset, verbose=False, return_images=True, max_examples=num_examples,
                                          **kwargs)
        noisy = noisy.reshape([num_examples*self.input_size[0], self.input_size[1], -1])
        gt = gt.reshape([num_examples*self.input_size[0], self.input_size[1], -1])
        pred = pred.reshape([num_examples*self.input_size[0], self.input_size[1], -1])
        image = np.concatenate([noisy, gt, pred], axis=1)

        image = cv2.cvtColor(to_int(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, 'epoch_{:03d}.jpg'.format(epoch)), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
