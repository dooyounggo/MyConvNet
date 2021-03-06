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
from models.vggnet import VGG16


class Unprocessing(ConvNet):
    def _init_model(self, **kwargs):
        dtypes = (tf.float32, tf.float32)
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        self._init_unprocessing(**kwargs)
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

                        _, bayer_img, noisy_img, variance, metadata = tf.map_fn(self.unprocess_images, self.X,
                                                                                dtype=(tf.float32, tf.float32,
                                                                                       tf.float32, tf.float32,
                                                                                       [tf.float32, tf.float32,
                                                                                        tf.float32, tf.float32,
                                                                                        tf.float32, tf.float32]),
                                                                                parallel_iterations=32, back_prop=False)

                        self.Y = process.process(bayer_img, metadata[2], metadata[3], metadata[0],
                                                 simple=self.simple_unprocessing)
                        self.Y.set_shape([None] + list(self.input_size))
                        vgg_input_gt[device] = self.Y

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

                        with tf.name_scope('nn') if self.model_scope is None else tf.name_scope(self.model_scope):
                            self.d = self._build_model()
                        if self.dtype is not tf.float32:
                            with tf.name_scope('{}/cast/'.format(self.compute_device + '_' + str(i))):
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        if self.channel_first:
                            self.d['pred'] = tf.transpose(self.d['pred'], perm=[0, 2, 3, 1])
                        tf.get_variable_scope().reuse_variables()

                        self.dicts.append(self.d)
                        self.pred = process.process(self.d['pred'], metadata[2], metadata[3], metadata[0],
                                                    simple=self.simple_unprocessing)
                        self.pred.set_shape([None] + list(self.input_size))
                        self.preds.append(self.pred)
                        vgg_input_pred[device] = self.pred

                        self.losses.append(self._build_loss(**kwargs))

        self._build_perceptual_loss(vgg_input_gt, vgg_input_pred, **kwargs)
        self._make_debug_images()

    def _build_loss(self, **kwargs):
        with tf.variable_scope('loss'):
            loss = self._loss_fn(**kwargs)
        return loss

    def _loss_fn(self, **kwargs):
        l2_loss_factor = kwargs.get('l2_loss_factor', 0.0)
        edge_loss_l1_factor = kwargs.get('edge_loss_l1_factor', 0.0)
        edge_loss_l2_factor = kwargs.get('edge_loss_l2_factor', 0.0)
        eps = 1e-8

        loss = tf.losses.absolute_difference(self.Y, self.pred)
        self.l1_losses.append(loss)
        if l2_loss_factor > 0.0:
            l2_loss = l2_loss_factor*tf.math.sqrt(tf.losses.mean_squared_error(self.Y, self.pred) + eps)
            loss += l2_loss
            self.l2_losses.append(l2_loss)
        else:
            self.l2_losses.append(0.0)

        with tf.variable_scope('edge_loss'):
            mask = tf.ones([1, self.input_size[0] - 2, self.input_size[1] - 2, 1])
            mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=0.0)  # Mask distorted boundaries

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
            edge_l2 = tf.math.sqrt(edge_l2 + eps)
            edge_l1_loss = edge_l1*edge_loss_l1_factor
            edge_l2_loss = edge_l2*edge_loss_l2_factor
            loss += edge_l1_loss + edge_l2_loss
            self.edge_l1_losses.append(edge_l1_loss)
            self.edge_l2_losses.append(edge_l2_loss)

        l1_factor = kwargs.get('l1_reg', 0e-8)
        l2_factor = kwargs.get('l2_reg', 1e-4)
        variables = self.get_collection('weight_variables')
        if kwargs.get('bias_norm_decay', False):
            variables += self.get_collection('bias_variables') + self.get_collection('norm_variables')
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
        self.Y_edges = []
        self.pred_edges = []
        self.l1_losses = []
        self.l2_losses = []
        self.edge_l1_losses = []
        self.edge_l2_losses = []
        self.perceptual_losses = []

        self.simple_unprocessing = kwargs.get('simple_unprocessing', False)
        self.add_unprocessing_noise = kwargs.get('add_unprocessing_noise', True)

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

        with tf.device(self.param_device):
            with tf.variable_scope('conditions/'):
                self._shot_noise = tf.placeholder(dtype=tf.float32, shape=[], name='shot_noise')
                self._read_noise = tf.placeholder(dtype=tf.float32, shape=[], name='read_noise')
        self.custom_feed_dict.update({self._shot_noise: shot_noise, self._read_noise: read_noise})

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

    def _init_vgg_net(self, **kwargs):
        class DummyNet(object):
            def __init__(self, is_train, monte_carlo, augmentation, total_steps):
                self.is_train = is_train
                self.monte_carlo = monte_carlo
                self.augmentation = augmentation
                self.total_steps = total_steps
        loss_factor = kwargs.get('perceptual_loss_factor', 0.0)
        kwargs['blocks_to_train'] = []
        if loss_factor > 0.0:
            dummynet = DummyNet(False, False, False, self.total_steps)
            self.vggnet_gt = VGG16(input_shape=self.input_size, num_classes=0, session=self.session,
                                   model_scope='vgg_gt', companion_networks={'DummyNet': dummynet},
                                   next_elements=self.next_elements, backbone_only=True, auto_build=False, **kwargs)
            self.vggnet_pred = VGG16(input_shape=self.input_size, num_classes=0, session=self.session,
                                     model_scope='vgg_pred', companion_networks={'DummyNet': dummynet},
                                     next_elements=self.next_elements, backbone_only=True, auto_build=False, **kwargs)
        else:
            self.vggnet_gt = None
            self.vggnet_pred = None

    def _build_perceptual_loss(self, vgg_input_gt, vgg_input_pred, **kwargs):
        loss_factor = kwargs.get('perceptual_loss_factor', 0.0)
        if loss_factor > 0.0:
            vggnet = self.vggnet_gt
            vgg_features_gt = []
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                device = '/{}:'.format(self.compute_device) + str(i)
                vggnet.next_elements[device][0] = vgg_input_gt[device]
            vggnet.build()
            for n in range(self.num_devices):
                vgg_features_gt.append(vggnet.dicts[n]['conv2_2'])

            vggnet = self.vggnet_pred
            vgg_features_pred = []
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                device = '/{}:'.format(self.compute_device) + str(i)
                vggnet.next_elements[device][0] = vgg_input_pred[device]
            vggnet.build()
            for n in range(self.num_devices):
                vgg_features_pred.append(vggnet.dicts[n]['conv2_2'])

            n = 0
            for i in range(self.device_offset, self.num_devices + self.device_offset):
                device = '/{}:'.format(self.compute_device) + str(i)
                with tf.device(device):
                    with tf.name_scope(self.compute_device + '_' + str(i) + '/'):
                        with tf.variable_scope('perceptual_loss'):
                            loss = loss_factor*tf.losses.mean_squared_error(vgg_features_gt[n], vgg_features_pred[n])
                            self.losses[n] += loss
                            self.perceptual_losses.append(loss)
                            n += 1
        else:
            for n in range(self.num_devices):
                self.perceptual_losses.append(0.0)

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

                self.debug_values.append(tf.reduce_mean(self.l1_losses, name='l1_loss'))
                self.debug_values.append(tf.reduce_mean(self.l2_losses, name='l2_loss'))
                self.debug_values.append(tf.reduce_mean(self.edge_l1_losses, name='edge_l1_loss'))
                self.debug_values.append(tf.reduce_mean(self.edge_l2_losses, name='edge_l2_loss'))
                self.debug_values.append(tf.reduce_mean(self.perceptual_losses, name='perceptual_loss'))

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

            if self.simple_unprocessing:
                # Inverts gamma compression.
                image = unprocess.gamma_expansion(image)
                # Inverts color correction.
                image = unprocess.apply_ccm(image, rgb2cam)
            else:
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
        if not self.add_unprocessing_noise:
            shot_noise = 0.0
            read_noise = 0.0
        shot_noise = tf.cond(self.is_train,
                             true_fn=lambda: shot_noise,
                             false_fn=lambda: tf.where(tf.math.is_nan(self._shot_noise), shot_noise, self._shot_noise))
        read_noise = tf.cond(self.is_train,
                             true_fn=lambda: read_noise,
                             false_fn=lambda: tf.where(tf.math.is_nan(self._read_noise), read_noise, self._read_noise))
        if self.add_unprocessing_noise:
            noisy_img = unprocess.add_noise(bayer_image, shot_noise, read_noise)
        else:
            noisy_img = bayer_image
        variance = shot_noise*noisy_img + read_noise

        metadata = [v for v in metadata.values()] + [shot_noise, read_noise]
        return image, bayer_image, noisy_img, variance, metadata

    def save_results(self, dataset, save_dir, epoch, max_examples=None, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        examples_per_image = kwargs.get('num_examples_per_image', 4)
        if max_examples is None:
            num_examples = min(16, dataset.num_examples)
        else:
            num_examples = min(max_examples, dataset.num_examples)
        num_images = num_examples//examples_per_image

        shot_noise = kwargs.get('shot_noise')
        read_noise = kwargs.get('read_noise')
        if shot_noise is None:
            shot_noise = np.nan
        if read_noise is None:
            read_noise = np.nan
        # FIXME: High noise level
        self.custom_feed_dict[self._shot_noise] = shot_noise*10**0.4
        self.custom_feed_dict[self._read_noise] = read_noise*10**0.8
        noisy, gt, pred, _ = self.predict(dataset, verbose=False, return_images=True, max_examples=num_examples,
                                          **kwargs)
        for i in range(num_images):
            sidx = i*examples_per_image
            eidx = (i + 1)*examples_per_image
            noisy_img = np.reshape(noisy[sidx:eidx], [examples_per_image*self.input_size[0], self.input_size[1], -1])
            gt_img = np.reshape(gt[sidx:eidx], [examples_per_image*self.input_size[0], self.input_size[1], -1])
            pred_img = np.reshape(pred[sidx:eidx], [examples_per_image*self.input_size[0], self.input_size[1], -1])
            image = np.concatenate([noisy_img, gt_img, pred_img], axis=1)

            image = to_int(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, 'epoch_{:03d}_{:03d}.jpg'.format(epoch, i)), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        # FIXME: Low noise level
        self.custom_feed_dict[self._shot_noise] = shot_noise*10**(-0.4)
        self.custom_feed_dict[self._read_noise] = read_noise*10**(-0.8)
        noisy, gt, pred, _ = self.predict(dataset, verbose=False, return_images=True, max_examples=num_examples,
                                          **kwargs)
        for i in range(num_images):
            sidx = i*examples_per_image
            eidx = (i + 1)*examples_per_image
            noisy_img = np.reshape(noisy[sidx:eidx],
                                   [examples_per_image*self.input_size[0], self.input_size[1], -1])
            gt_img = np.reshape(gt[sidx:eidx], [examples_per_image*self.input_size[0], self.input_size[1], -1])
            pred_img = np.reshape(pred[sidx:eidx],
                                  [examples_per_image*self.input_size[0], self.input_size[1], -1])
            image = np.concatenate([noisy_img, gt_img, pred_img], axis=1)

            image = to_int(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, 'epoch_{:03d}_{:03d}.jpg'.format(epoch, i + num_images)), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        self.custom_feed_dict[self._shot_noise] = shot_noise
        self.custom_feed_dict[self._read_noise] = read_noise
