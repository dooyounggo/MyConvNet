"""
Build convolutional neural networks using TensorFlow low-level APIs.
"""

import time
import warnings
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from contextlib import nullcontext


class ConvNet(object):
    def __init__(self, input_shape, num_classes, loss_weights=None, **kwargs):
        """
        :param input_shape: list or tuple, network input size.
        :param num_classes: int, number of classes.
        :param loss_weights: list or tuple, weighting factors for softmax losses.
        :param kwargs: dict, extra arguments containing hyperparameters.
        """
        graph = tf.get_default_graph()
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = kwargs.get('num_parallel_calls', 0)
        config.inter_op_parallelism_threads = 0
        config.gpu_options.force_gpu_compatible = True
        config.allow_soft_placement = False
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7  # FIXME
        self._session = tf.Session(graph=graph, config=config)  # TF main session
        self._top_scope = tf.get_variable_scope()

        self._input_size = input_shape  # Size of the network input (i.e., the first convolution layer).
        self._num_classes = num_classes
        self._loss_weights = loss_weights  # Weight values for the softmax losses of each class

        self._dtype = tf.float16 if kwargs.get('half_precision', False) else tf.float32
        self._channel_first = kwargs.get('channel_first', False)
        self._argmax_output = kwargs.get('argmax_output', False)
        self._num_gpus = kwargs.get('num_gpus', 1)
        self._cpu_offset = kwargs.get('cpu_offset', 0)
        self._gpu_offset = kwargs.get('gpu_offset', 0)

        self._param_device = '/cpu:{}'.format(self.cpu_offset)

        self._padded_size = np.round(np.array(self.input_size[0:2])*(1.0 + kwargs.get('zero_pad_ratio', 0.0)))
        self.pad_value = kwargs.get('pad_value', 0.5)

        self._dropout_weights = kwargs.get('dropout_weights', False)
        self._dropout_features = kwargs.get('dropout_features', True)

        self._blocks_to_train = kwargs.get('blocks_to_train', None)
        self._update_batch_norm = kwargs.get('update_batch_norm', True)

        self._moving_average_decay = kwargs.get('moving_average_decay', 0.99)
        self._batch_norm_decay = kwargs.get('batch_norm_decay', 0.99)

        self._feature_reduction = kwargs.get('feature_reduction_factor', 0)

        self.handles = []
        self.X_in = []
        self.Y_in = []
        self.Xs = []
        self.Ys = []
        self.preds = []
        self.valid_masks = []
        self.losses = []
        self.gcams = []
        self.bytes_in_use = []

        self._flops = 0
        self._params = 0

        self.backbone_only = False
        self.dicts = []
        self._update_ops = []

        with tf.device(self.param_device):
            with tf.variable_scope('conditions'):
                self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')
                self.monte_carlo = tf.placeholder(tf.bool, shape=[], name='monte_carlo')
                self.augmentation = tf.placeholder(tf.bool, shape=[], name='augmentation')
                self.total_steps = tf.placeholder(tf.int64, shape=[], name='total_steps')

            with tf.variable_scope('calc'):
                self.global_step = tf.train.get_or_create_global_step()
                global_step = tf.cast(self.global_step, dtype=tf.float32)

                self.dropout_rate = tf.cond(tf.math.logical_or(self.is_train, self.monte_carlo),
                                            lambda: tf.constant(kwargs.get('dropout_rate', 0.0), dtype=self.dtype),
                                            lambda: tf.constant(0.0, dtype=self.dtype, name='0'),
                                            name='dropout_rate')

                if self.dropout_weights:
                    self.dropout_rate_weights = self.dropout_rate
                else:
                    self.dropout_rate_weights = tf.constant(0.0, dtype=self.dtype, name='0')
                if self.dropout_features:
                    self.dropout_rate_features = self.dropout_rate
                else:
                    self.dropout_rate_features = tf.constant(0.0, dtype=self.dtype, name='0')
                if kwargs.get('zero_center', True):
                    self.image_mean = tf.constant(kwargs.get('image_mean', 0.5), dtype=tf.float32, name='image_mean')
                else:
                    self.image_mean = tf.constant(0.0, dtype=tf.float32, name='0')

                self.linear_schedule_multiplier = global_step/tf.cast(self.total_steps, dtype=tf.float32)

                self._dummy_image = tf.zeros([4, 8, 8, 3], dtype=tf.float32, name='dummy_image')

        self.ema = tf.train.ExponentialMovingAverage(decay=self.moving_average_decay)

        self.debug_value = self.linear_schedule_multiplier
        self.debug_images_0 = self._dummy_image
        self.debug_images_1 = self._dummy_image

        self._init_params()
        self._init_model(**kwargs)

        if self.argmax_output:
            with tf.device(self.param_device):
                with tf.variable_scope('calc/'):
                    valid_mask = tf.cast(self.valid_mask, dtype=tf.int32)
                    invalid_mask = tf.cast(tf.logical_not(self.valid_mask), dtype=tf.int32)
                    self.Y_all = tf.math.argmax(self.Y_all, axis=-1, output_type=tf.int32)*valid_mask - invalid_mask
                    self.Y_all = self.Y_all[..., tf.newaxis]
                    self.pred = tf.math.argmax(self.pred, axis=-1, output_type=tf.int32)*valid_mask - invalid_mask
                    self.pred = self.pred[..., tf.newaxis]

        print('\nNumber of GPUs : {}'.format(self._num_gpus))
        print('Total number of units: {}'.format(self._num_blocks))
        print('\n# FLOPs : {:-15,}\n# Params: {:-15,}\n'.format(int(self._flops), int(self._params)))

    def _init_params(self):
        """
        Parameter initialization.
        Initialize model parameters.
        """
        pass

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        :return dict containing tensors. Must include 'logits' and 'pred' tensors.
        """
        pass

    @property
    def session(self):
        return self._session

    @property
    def top_scope(self):
        return self._top_scope

    @property
    def input_size(self):
        return self._input_size

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def loss_weights(self):
        return self._loss_weights

    @property
    def dtype(self):
        return self._dtype

    @property
    def channel_first(self):
        return self._channel_first

    @property
    def argmax_output(self):
        return self._argmax_output

    @property
    def num_gpus(self):
        return self._num_gpus

    @property
    def cpu_offset(self):
        return self._cpu_offset

    @property
    def gpu_offset(self):
        return self._gpu_offset

    @property
    def param_device(self):
        return self._param_device

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def dropout_weights(self):
        return self._dropout_weights

    @property
    def dropout_features(self):
        return self._dropout_features

    @property
    def blocks_to_train(self):
        return self._blocks_to_train

    @property
    def update_batch_norm(self):
        return self._update_batch_norm

    @property
    def moving_average_decay(self):
        return self._moving_average_decay

    @property
    def batch_norm_decay(self):
        return self._batch_norm_decay

    @property
    def feature_reduction(self):
        return self._feature_reduction

    @property
    def update_ops(self):
        return self._update_ops

    def close(self):
        self.session.close()

    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         [None])
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.gpu_offset, self.num_gpus + self.gpu_offset):
                self._curr_device = i
                self._curr_block = 0
                self._num_blocks = 1
                self._curr_dependent_op = 0
                with tf.device('/gpu:' + str(i)):
                    with tf.name_scope('gpu{}'.format(i)):
                        handle = tf.placeholder(tf.string, shape=[], name='handle')  # A handle for feedable iterator
                        self.handles.append(handle)
                        iterator = tf.data.Iterator.from_string_handle(handle, (tf.float32, tf.float32),
                                                                       output_shapes=output_shapes)
                        self.X, self.Y = iterator.get_next()
                        self.X_in.append(self.X)
                        self.Y_in.append(self.Y)

                        # FIXME: Fake label generation
                        self.Y = tf.where(tf.is_nan(self.Y),  # Fake label is created when the label is NaN
                                          0.0 - tf.ones_like(self.Y, dtype=tf.float32),
                                          self.Y)

                        self.Y = tf.cast(self.Y, dtype=tf.int32)
                        self.Y = tf.one_hot(self.Y, depth=self.num_classes, dtype=tf.float32)  # one-hot encoding

                        if self._padded_size[0] > self.input_size[0] or self._padded_size[1] > self.input_size[1]:
                            self.X = self.zero_pad(self.X, pad_value=self.pad_value)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        self.X = tf.cond(self.augmentation,
                                         lambda: self.augment_images(self.X, **kwargs),
                                         lambda: self.center_crop(self.X),
                                         name='augmentation')
                        if kwargs.get('cutmix', False):
                            self._cutmix_scheduling = kwargs.get('cutmix_scheduling', False)
                            self.X, self.Y = tf.cond(self.is_train,
                                                     lambda: self.cutmix(self.X, self.Y),
                                                     lambda: (self.X, self.Y),
                                                     name='cutmix')
                        self.Xs.append(self.X)
                        self.Ys.append(self.Y)

                        self.X *= 2  # Set input range in [-1, 1]

                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])

                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.X = tf.cast(self.X, dtype=self.dtype)
                        with tf.name_scope('nn'):
                            self.d = self._build_model(**kwargs)
                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.d['logits'] = tf.cast(self.d['logits'], dtype=tf.float32)
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        tf.get_variable_scope().reuse_variables()

                        self.dicts.append(self.d)

                        if 'block_{}'.format(self.num_blocks - 1) in self.d:
                            self.gcams.append(self.grad_cam(self.d['logits'],
                                                            self.d['block_{}'.format(self.num_blocks - 1)],
                                                            y=None))
                        else:
                            self.gcams.append(self.X)

                        self.logits = self.d['logits']
                        self.pred = self.d['pred']
                        self.preds.append(self.pred)
                        self.losses.append(self._build_loss(**kwargs))

                        self.bytes_in_use.append(tf.contrib.memory_stats.BytesInUse())

        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                self.X_all = tf.concat(self.Xs, axis=0, name='x') + self.image_mean
                self.Y_all = tf.concat(self.Ys, axis=0, name='y_true')
                self.pred = tf.concat(self.preds, axis=0, name='y_pred')
                self.valid_mask = tf.concat(self.valid_masks, axis=0, name='valid_mask')
                self.loss = tf.reduce_mean(self.losses, name='mean_loss')
                self.gcam = tf.concat(self.gcams, axis=0, name='grad_cam')

                self.input_images = tf.concat(self.X_in, axis=0, name='x_in')
                self.debug_images_0 = tf.clip_by_value(self.gcam/2 + self.X_all, 0, 1)
                self.debug_images_1 = tf.clip_by_value(self.gcam*self.X_all, 0, 1)

    def _build_loss(self, **kwargs):
        l1_factor = kwargs.get('l1_reg', 0e-8)
        l2_factor = kwargs.get('l2_reg', 1e-4)
        ls_factor = kwargs.get('label_smoothing', 0.0)
        focal_loss_factor = kwargs.get('focal_loss_factor', 0.0)
        sigmoid_focal_loss_factor = kwargs.get('sigmoid_focal_loss_factor', 0.0)

        all_vars = tf.get_collection('weight_variables')
        if kwargs.get('bias_norm_decay', False):
            all_vars += tf.get_collection('bias_variables') + tf.get_collection('norm_variables')
        variables = [var for var in all_vars if var.device[-1] == str(self._curr_device)]
        valid_eps = 1e-5

        w = self.loss_weights
        if w is None:
            w = np.ones(self.num_classes, dtype=np.float32)
        else:
            w = np.array(w, dtype=np.float32)
        print('\nLoss weights: ', w)

        with tf.variable_scope('loss'):
            w = tf.constant(w, dtype=tf.float32, name='class_weights')
            w = tf.expand_dims(w, axis=0)
            while len(w.get_shape()) < len(self.Y.get_shape()):
                w = tf.expand_dims(w, axis=1)
            batch_weights = tf.reduce_sum(self.Y*w, axis=-1)

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

            with tf.variable_scope('valid_mask'):
                sumval = tf.reduce_sum(self.Y, axis=-1)
                valid_g = tf.greater(sumval, 1.0 - valid_eps)
                valid_l = tf.less(sumval, 1.0 + valid_eps)
                valid_mask = tf.logical_and(valid_g, valid_l)
                self.valid_masks.append(valid_mask)
                valid_mask = tf.cast(valid_mask, dtype=tf.float32)

            if ls_factor > 0.0:
                labels = self.label_smoothing(ls_factor)
            else:
                labels = self.Y

            softmax_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits, axis=-1)
            if focal_loss_factor > 0.0:
                gamma = focal_loss_factor
                with tf.variable_scope('focal_loss'):
                    pred = tf.reduce_sum(self.Y*self.pred, axis=-1)
                    focal_loss = tf.pow(1.0 - pred, gamma)
                    softmax_losses *= focal_loss
            if sigmoid_focal_loss_factor > 0.0:
                alpha = sigmoid_focal_loss_factor
                with tf.variable_scope('sigmoid_focal_loss'):
                    pred = tf.reduce_sum(self.Y*self.pred, axis=-1)
                    sigmoid_focal_loss = tf.stop_gradient(1.0 - tf.math.sigmoid(alpha*(pred - 0.5)))
                    sigmoid_focal_loss /= 1.0 - tf.math.sigmoid(-0.5*alpha)
                    softmax_losses *= sigmoid_focal_loss
            softmax_loss = tf.reduce_mean(batch_weights*valid_mask*softmax_losses)

            loss = softmax_loss + l1_reg_loss + l2_reg_loss

        return loss

    def label_smoothing(self, ls_factor, name='label_smoothing'):
        with tf.variable_scope(name):
            ls_factor = tf.constant(ls_factor, dtype=tf.float32, name='label_smoothing_factor')
            labels = self.Y*(1.0 - ls_factor) + ls_factor/self.num_classes

        return labels

    def predict(self, dataset, verbose=False, return_images=True, **kwargs):
        batch_size = dataset.batch_size
        augment_test = kwargs.get('augment_test', False)

        pred_size = dataset.num_examples
        num_steps = np.ceil(pred_size/batch_size).astype(int)
        monte_carlo = kwargs.get('monte_carlo', False)

        dataset.initialize(self.session)
        handles = dataset.get_string_handles(self.session)
        self.session.run(tf.get_collection('var_assign_ops'))  # Assign main variables to device variables

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
        _Y_true = np.zeros([pred_size] + self.pred.get_shape().as_list()[1:], dtype=np.float32)
        _Y_pred = np.zeros([pred_size] + self.pred.get_shape().as_list()[1:], dtype=np.float32)
        _loss_pred = np.zeros(pred_size, dtype=np.float32)
        start_time = time.time()
        for i in range(num_steps):
            try:
                X, Y_true, Y_pred, loss_pred = self.session.run([self.X_all, self.Y_all, self.pred, self.loss],
                                                                feed_dict=feed_dict)
                sidx = i*batch_size
                eidx = (i + 1)*batch_size
                if return_images:
                    _X[sidx:eidx] = X
                _Y_true[sidx:eidx] = Y_true
                _Y_pred[sidx:eidx] = Y_pred
                _loss_pred[sidx:eidx] = loss_pred
            except tf.errors.OutOfRangeError:
                if verbose:
                    print('The last iteration ({} data) has been ignored'.format(pred_size - i*batch_size))

        if verbose:
            print('Total prediction time: {:.2f} sec'.format(time.time() - start_time))

        _loss_pred = np.mean(_loss_pred, axis=0)

        return _X, _Y_true, _Y_pred, _loss_pred

    def features(self, dataset, tensors, **kwargs):  # Return any deep features
        batch_size = dataset.batch_size
        augment_test = kwargs.get('augment_test', False)

        pred_size = dataset.num_examples
        num_steps = np.ceil(pred_size/batch_size).astype(int)
        monte_carlo = kwargs.get('monte_carlo', False)

        dataset.initialize(self.session)
        handles = dataset.get_string_handles(self.session)
        self.session.run(tf.get_collection('var_assign_ops'))  # Assign main variables to device variables

        feed_dict = {self.is_train: False,
                     self.monte_carlo: monte_carlo,
                     self.augmentation: augment_test}
        for h_t, h in zip(self.handles, handles):
            feed_dict.update({h_t: h})

        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        batched_features = []
        for i in range(num_steps):
            feat = self.session.run(tensors, feed_dict=feed_dict)
            batched_features.append(feat)

        features = []
        for feat in zip(*batched_features):
            features.append(np.concatenate(feat, axis=0))

        return features

    def zero_pad(self, x, pad_value=0.0):
        with tf.variable_scope('zero_pad'):
            shape_tensor = tf.cast(tf.shape(x), dtype=tf.float32)
            h = shape_tensor[1]
            w = shape_tensor[2]
            pad_h = tf.maximum(self._padded_size[0] - h, 0.0)
            pad_w = tf.maximum(self._padded_size[1] - w, 0.0)
            paddings = [[0, 0],
                        [tf.cast(tf.floor(pad_h/2), dtype=tf.int32), tf.cast(tf.ceil(pad_h/2), dtype=tf.int32)],
                        [tf.cast(tf.floor(pad_w/2), dtype=tf.int32), tf.cast(tf.ceil(pad_w/2), dtype=tf.int32)],
                        [0, 0]]
            x = tf.pad(x, paddings, constant_values=pad_value)

        return x

    def augment_images(self, x, mask=None, **kwargs):
        rand_blur = kwargs.get('rand_blur_stddev', 0.0) > 0.0
        rand_affine = kwargs.get('rand_affine', False)
        rand_crop = kwargs.get('rand_crop', False)
        rand_distortion = kwargs.get('rand_distortion', False)

        if rand_blur:
            x = self.gaussian_blur(x, **kwargs)

        if rand_affine:
            x, mask = self.affine_augment(x, mask, **kwargs)

        if rand_crop:
            x, mask = self.rand_crop(x, mask, **kwargs)
        else:
            x = self.center_crop(x)
            if mask is not None:
                mask = self.center_crop(mask)

        if rand_distortion:
            x = self.rand_hue(x, **kwargs)
            x = self.rand_saturation(x, **kwargs)
            x = self.rand_color_balance(x, **kwargs)
            x = self.rand_equalization(x, **kwargs)
            x = self.rand_contrast(x, **kwargs)
            x = self.rand_brightness(x, **kwargs)
            x = self.rand_noise(x, **kwargs)
            x = tf.clip_by_value(x, 0.0 - self.image_mean, 1.0 - self.image_mean)
            x = self.rand_solarization(x, **kwargs)
            x = self.rand_posterization(x, **kwargs)

        if mask is None:
            return x
        else:
            return x, mask

    def gaussian_blur(self, x, **kwargs):
        with tf.variable_scope('gaussian_blur'):
            max_stddev = kwargs.get('rand_blur_stddev', 0.0)
            scheduling = kwargs.get('rand_blur_scheduling', False)
            if scheduling > 0:
                max_stddev *= self.linear_schedule_multiplier
            elif scheduling < 0:
                max_stddev *= 1.0 - self.linear_schedule_multiplier

            column_base = -0.5*np.array([[7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)**2
            row_base = np.transpose(column_base)
            column_base = tf.tile(tf.constant(column_base[:, :, np.newaxis, np.newaxis], dtype=tf.float32),
                                  multiples=(1, 1, self.input_size[-1], 1))
            row_base = tf.tile(tf.constant(row_base[:, :, np.newaxis, np.newaxis], dtype=tf.float32),
                               multiples=(1, 1, self.input_size[-1], 1))
            pi = tf.constant(np.pi, dtype=tf.float32)

            var = tf.random.uniform([], minval=0.0, maxval=max_stddev, dtype=tf.float32)**2
            denom = tf.math.sqrt(2*pi*var)
            h_filt = tf.math.exp(column_base/var)/denom
            w_filt = tf.math.exp(row_base/var)/denom

            x = tf.nn.depthwise_conv2d(x, h_filt, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.depthwise_conv2d(x, w_filt, strides=[1, 1, 1, 1], padding='SAME')

        return x

    def affine_augment(self, x, mask=None, **kwargs):  # Scale, ratio, translation, rotation, shear, and reflection
        scheduling = kwargs.get('rand_affine_scheduling', False)
        with tf.variable_scope('affine_augment'):
            shape_tensor = tf.shape(x)
            batch_size = shape_tensor[0]
            h = tf.cast(shape_tensor[1], dtype=tf.float32)
            w = tf.cast(shape_tensor[2], dtype=tf.float32)

            lower, upper = kwargs.get('rand_scale', (1.0, 1.0))
            if scheduling > 0:
                lower = 1.0 - (1.0 - lower)*self.linear_schedule_multiplier
                upper = 1.0 - (1.0 - upper)*self.linear_schedule_multiplier
            elif scheduling < 0:
                lower = 1.0 - (1.0 - lower)*(1.0 - self.linear_schedule_multiplier)
                upper = 1.0 - (1.0 - upper)*(1.0 - self.linear_schedule_multiplier)
            # base = upper/lower
            # randvals = tf.random.uniform([batch_size, 1], dtype=tf.float32)
            # rand_scale = lower*tf.math.pow(base, randvals)
            rand_scale = tf.random.uniform([], lower, upper, dtype=tf.float32)

            lower, upper = kwargs.get('rand_ratio', (1.0, 1.0))
            if scheduling > 0:
                lower = tf.math.pow(lower, self.linear_schedule_multiplier)
                upper = tf.math.pow(upper, self.linear_schedule_multiplier)
            elif scheduling < 0:
                lower = tf.math.pow(lower, 1.0 - self.linear_schedule_multiplier)
                upper = tf.math.pow(upper, 1.0 - self.linear_schedule_multiplier)
            base = upper/lower
            randvals = tf.random.uniform([batch_size, 1], dtype=tf.float32)
            rand_ratio = lower*tf.math.pow(base, randvals)

            rand_x_scale = tf.math.sqrt(rand_scale*rand_ratio)
            rand_y_scale = tf.math.sqrt(rand_scale/rand_ratio)

            val = kwargs.get('rand_rotation', 0)
            if scheduling > 0:
                val *= self.linear_schedule_multiplier
            elif scheduling < 0:
                val *= 1.0 - self.linear_schedule_multiplier
            rand_rotation = (tf.random.uniform([batch_size, 1]) - 0.5)*val*(np.pi/180)

            val = kwargs.get('rand_shear', 0)
            if scheduling > 0:
                val *= self.linear_schedule_multiplier
            elif scheduling < 0:
                val *= 1.0 - self.linear_schedule_multiplier
            rand_shear = (tf.random.uniform([batch_size, 1]) - 0.5)*val*(np.pi/180)

            val = kwargs.get('rand_x_trans', 0)
            if scheduling > 0:
                val *= self.linear_schedule_multiplier
            elif scheduling < 0:
                val *= 1.0 - self.linear_schedule_multiplier
            rand_x_trans = (tf.random.uniform([batch_size, 1]) - 0.5)*val*w \
                           + 0.5*w*(1.0 - rand_x_scale*tf.math.cos(rand_rotation)) \
                           + 0.5*h*rand_y_scale*tf.math.sin(rand_rotation + rand_shear)

            val = kwargs.get('rand_y_trans', 0)
            if scheduling > 0:
                val *= self.linear_schedule_multiplier
            elif scheduling < 0:
                val *= 1.0 - self.linear_schedule_multiplier
            rand_y_trans = (tf.random.uniform([batch_size, 1]) - 0.5)*val*h \
                           - 0.5*w*rand_x_scale*tf.math.sin(rand_rotation) \
                           + 0.5*h*(1.0 - rand_y_scale*tf.math.cos(rand_rotation + rand_shear))

            a0a = rand_x_scale*tf.math.cos(rand_rotation + rand_shear)
            a1a = -rand_y_scale*tf.math.sin(rand_rotation)
            a2a = rand_x_trans
            b0a = rand_x_scale*tf.math.sin(rand_rotation + rand_shear)
            b1a = rand_y_scale*tf.math.cos(rand_rotation)
            b2a = rand_y_trans

            val = kwargs.get('rand_x_reflect', True)
            if scheduling > 0:
                val *= self.linear_schedule_multiplier
            elif scheduling < 0:
                val *= 1.0 - self.linear_schedule_multiplier
            rand_x_reflect = tf.math.round(tf.random.uniform([batch_size, 1])*val)

            val = kwargs.get('rand_y_reflect', False)
            if scheduling > 0:
                val *= self.linear_schedule_multiplier
            elif scheduling < 0:
                val *= 1.0 - self.linear_schedule_multiplier
            rand_y_reflect = tf.math.round(tf.random.uniform([batch_size, 1])*val)

            a0r = 1.0 - 2.0*rand_x_reflect
            # a1r = tf.zeros([batch_size, 1], dtype=tf.float32)
            a2r = rand_x_reflect*w
            # b0r = tf.zeros([batch_size, 1], dtype=tf.float32)
            b1r = 1.0 - 2.0*rand_y_reflect
            b2r = rand_y_reflect*h

            a0 = a0a*a0r
            a1 = a1a*a0r
            a2 = a2a*a0r + a2r
            b0 = b0a*b1r
            b1 = b1a*b1r
            b2 = b2a*b1r + b2r
            c0 = tf.zeros([batch_size, 1], dtype=tf.float32)
            c1 = tf.zeros([batch_size, 1], dtype=tf.float32)
            transforms = tf.concat([a0, a1, a2, b0, b1, b2, c0, c1], axis=1)

            x = tf.contrib.image.transform(x, transforms, interpolation='BILINEAR')
            if mask is not None:
                mask = tf.contrib.image.transform(mask, transforms, interpolation='NEAREST')

        return x, mask

    def rand_crop(self, x, mask=None, **kwargs):
        with tf.variable_scope('rand_crop'):
            self._crop_scale = kwargs.get('rand_crop_scale', (1.0, 1.0))  # Size of crop windows
            self._crop_ratio = kwargs.get('rand_crop_ratio', (1.0, 1.0))  # Aspect ratio of crop windows
            self._extend_bbox_index_range = kwargs.get('extend_bbox_index_range', False)
            self._min_object_size = kwargs.get('min_object_size', None)
            self._interpolation = kwargs.get('resize_interpolation', 'bilinear')  # Interpolation method
            self._rand_interpolation = kwargs.get('rand_interpolation', True)
            self._crop_scheduling = kwargs.get('rand_crop_scheduling', False)
            if mask is None:
                x = tf.map_fn(self.rand_crop_image, x, parallel_iterations=32, back_prop=False)
            else:
                x, mask = tf.map_fn(self.rand_crop_image_and_mask, (x, mask), dtype=(tf.float32, tf.float32),
                                    parallel_iterations=32, back_prop=False)

        return x, mask

    def rand_crop_image(self, x):
        image = x

        shape_tensor = tf.shape(image)
        h = tf.cast(shape_tensor[0], dtype=tf.int32)
        w = tf.cast(shape_tensor[1], dtype=tf.int32)

        lower, upper = self._crop_scale
        if self._crop_scheduling > 0:
            lower = 1.0 - (1.0 - lower)*self.linear_schedule_multiplier
            upper = 1.0 - (1.0 - upper)*self.linear_schedule_multiplier
        elif self._crop_scheduling < 0:
            lower = 1.0 - (1.0 - lower)*(1.0 - self.linear_schedule_multiplier)
            upper = 1.0 - (1.0 - upper)*(1.0 - self.linear_schedule_multiplier)
        scale_lower = lower
        # a = upper**2 - lower**2
        # b = lower**2
        # randval = tf.random.uniform([], dtype=tf.float32)
        # rand_scale = tf.math.sqrt(a*randval + b)
        rand_scale = tf.random.uniform([], lower, upper, dtype=tf.float32)

        lower, upper = self._crop_ratio
        if self._crop_scheduling > 0:
            lower = tf.math.pow(lower, self.linear_schedule_multiplier)
            upper = tf.math.pow(upper, self.linear_schedule_multiplier)
        elif self._crop_scheduling < 0:
            lower = tf.math.pow(lower, 1.0 - self.linear_schedule_multiplier)
            upper = tf.math.pow(upper, 1.0 - self.linear_schedule_multiplier)
        base = upper/lower
        randval = tf.random.uniform([], dtype=tf.float32)
        rand_ratio = lower*tf.math.pow(base, randval)

        rand_x_scale = tf.math.sqrt(rand_scale/rand_ratio)
        rand_y_scale = tf.math.sqrt(rand_scale*rand_ratio)

        size_h_full = tf.cast(tf.math.round(self.input_size[0]*rand_y_scale), dtype=tf.int32)
        size_w_full = tf.cast(tf.math.round(self.input_size[1]*rand_x_scale), dtype=tf.int32)
        size_h = tf.math.minimum(h, size_h_full)
        size_w = tf.math.minimum(w, size_w_full)

        offset_h = tf.random.uniform([], 0, h - size_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, w - size_w + 1, dtype=tf.int32)

        if self._extend_bbox_index_range:
            with tf.variable_scope('full_index_range'):
                offset_h_full = tf.random.uniform([], 0, h, dtype=tf.int32)
                offset_w_full = tf.random.uniform([], 0, w, dtype=tf.int32)
                x_min = offset_w_full - size_w_full//2
                x_max = tf.math.minimum(x_min + size_w_full, w)
                x_min = tf.math.maximum(0, x_min)
                y_min = offset_h_full - size_h_full//2
                y_max = tf.math.minimum(y_min + size_h_full, h)
                y_min = tf.math.maximum(0, y_min)

                crop_h = y_max - y_min
                crop_w = x_max - x_min
                if self._min_object_size is None:
                    min_object_size = scale_lower
                else:
                    min_object_size = self._min_object_size
                min_object_area = min_object_size*tf.cast(h, dtype=tf.float32)*tf.cast(w, dtype=tf.float32)
                output_ratio = self.input_size[1]/self.input_size[0]
                crop_area = tf.cast(crop_h*crop_w, dtype=tf.float32)
                crop_ratio = tf.cast(crop_h, dtype=tf.float32)/tf.cast(crop_w, dtype=tf.float32)*output_ratio

                valid_size = tf.math.greater_equal(crop_area, min_object_area)
                valid_ratio = tf.math.logical_and(tf.math.greater_equal(crop_ratio, self._crop_ratio[0]),
                                                  tf.math.less_equal(crop_ratio, self._crop_ratio[1]))
                valid = tf.math.logical_and(valid_size, valid_ratio)

            image = tf.cond(valid,  # Cropped image depends on whether the cropping in the full index range is valid
                            lambda: tf.expand_dims(tf.slice(image, [y_min, x_min, 0], [crop_h, crop_w, -1]), axis=0),
                            lambda: tf.expand_dims(tf.slice(image, [offset_h, offset_w, 0], [size_h, size_w, -1]),
                                                   axis=0))
        else:
            image = tf.expand_dims(tf.slice(image, [offset_h, offset_w, 0], [size_h, size_w, -1]), axis=0)

        re_size = self.input_size[0:2]
        if self._rand_interpolation:
            num = tf.random.uniform([], 0, 2, dtype=tf.int32)
            image = tf.cond(tf.cast(num, dtype=tf.bool),
                            lambda: tf.image.resize_nearest_neighbor(image, re_size, align_corners=True),
                            lambda: tf.image.resize_bilinear(image, re_size, align_corners=True))
        elif self._interpolation.lower() == 'nearest' or self._interpolation.lower() == 'nearest neighbor':
            image = tf.image.resize_nearest_neighbor(image, re_size, align_corners=True)
        elif self._interpolation.lower() == 'bilinear':
            image = tf.image.resize_bilinear(image, re_size, align_corners=True)
        elif self._interpolation.lower() == 'bicubic':
            warnings.warn('Bicubic interpolation is not supported for GPU. Bilinear is used instead.', UserWarning)
            image = tf.image.resize_bilinear(image, re_size, align_corners=True)
        else:
            raise(ValueError, 'Interpolation method of {} is not supported.'.format(self._interpolation))

        image = tf.reshape(image, self.input_size)

        return image

    def rand_crop_image_and_mask(self, x):
        image = x[0]
        mask = x[1]

        shape_tensor = tf.shape(image)
        h = tf.cast(shape_tensor[0], dtype=tf.int32)
        w = tf.cast(shape_tensor[1], dtype=tf.int32)

        lower, upper = self._crop_scale
        if self._crop_scheduling > 0:
            lower = 1.0 - (1.0 - lower)*self.linear_schedule_multiplier
            upper = 1.0 - (1.0 - upper)*self.linear_schedule_multiplier
        elif self._crop_scheduling < 0:
            lower = 1.0 - (1.0 - lower)*(1.0 - self.linear_schedule_multiplier)
            upper = 1.0 - (1.0 - upper)*(1.0 - self.linear_schedule_multiplier)
        scale_lower = lower
        # a = upper**2 - lower**2
        # b = lower**2
        # randval = tf.random.uniform([], dtype=tf.float32)
        # rand_scale = tf.math.sqrt(a*randval + b)
        rand_scale = tf.random.uniform([], lower, upper, dtype=tf.float32)

        lower, upper = self._crop_ratio
        if self._crop_scheduling > 0:
            lower = tf.math.pow(lower, self.linear_schedule_multiplier)
            upper = tf.math.pow(upper, self.linear_schedule_multiplier)
        elif self._crop_scheduling < 0:
            lower = tf.math.pow(lower, 1.0 - self.linear_schedule_multiplier)
            upper = tf.math.pow(upper, 1.0 - self.linear_schedule_multiplier)
        base = upper/lower
        randval = tf.random.uniform([], dtype=tf.float32)
        rand_ratio = lower*tf.math.pow(base, randval)

        rand_x_scale = tf.math.sqrt(rand_scale/rand_ratio)
        rand_y_scale = tf.math.sqrt(rand_scale*rand_ratio)

        size_h_full = tf.cast(tf.math.round(self.input_size[0]*rand_y_scale), dtype=tf.int32)
        size_w_full = tf.cast(tf.math.round(self.input_size[1]*rand_x_scale), dtype=tf.int32)
        size_h = tf.math.minimum(h, size_h_full)
        size_w = tf.math.minimum(w, size_w_full)

        offset_h = tf.random.uniform([], 0, h - size_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, w - size_w + 1, dtype=tf.int32)

        if self._extend_bbox_index_range:
            with tf.variable_scope('full_index_range'):
                offset_h_full = tf.random.uniform([], 0, h, dtype=tf.int32)
                offset_w_full = tf.random.uniform([], 0, w, dtype=tf.int32)
                x_min = offset_w_full - size_w_full//2
                x_max = tf.math.minimum(x_min + size_w_full, w)
                x_min = tf.math.maximum(0, x_min)
                y_min = offset_h_full - size_h_full//2
                y_max = tf.math.minimum(y_min + size_h_full, h)
                y_min = tf.math.maximum(0, y_min)

                crop_h = y_max - y_min
                crop_w = x_max - x_min
                if self._min_object_size is None:
                    min_object_size = scale_lower
                else:
                    min_object_size = self._min_object_size
                min_object_area = min_object_size*tf.cast(h, dtype=tf.float32)*tf.cast(w, dtype=tf.float32)
                output_ratio = self.input_size[1]/self.input_size[0]
                crop_area = tf.cast(crop_h*crop_w, dtype=tf.float32)
                crop_ratio = tf.cast(crop_h, dtype=tf.float32)/tf.cast(crop_w, dtype=tf.float32)*output_ratio

                valid_size = tf.math.greater_equal(crop_area, min_object_area)
                valid_ratio = tf.math.logical_and(tf.math.greater_equal(crop_ratio, self._crop_ratio[0]),
                                                  tf.math.less_equal(crop_ratio, self._crop_ratio[1]))
                valid = tf.math.logical_and(valid_size, valid_ratio)

            image = tf.cond(valid,  # Cropped image depends on whether the cropping in the full index range is valid
                            lambda: tf.expand_dims(tf.slice(image, [y_min, x_min, 0], [crop_h, crop_w, -1]), axis=0),
                            lambda: tf.expand_dims(tf.slice(image, [offset_h, offset_w, 0], [size_h, size_w, -1]),
                                                   axis=0))
            mask = tf.cond(valid,
                           lambda: tf.expand_dims(tf.slice(mask, [y_min, x_min, 0], [crop_h, crop_w, -1]), axis=0),
                           lambda: tf.expand_dims(tf.slice(mask, [offset_h, offset_w, 0], [size_h, size_w, -1]),
                                                  axis=0))
        else:
            image = tf.expand_dims(tf.slice(image, [offset_h, offset_w, 0], [size_h, size_w, -1]), axis=0)
            mask = tf.expand_dims(tf.slice(mask, [offset_h, offset_w, 0], [size_h, size_w, -1]), axis=0)

        re_size = self.input_size[0:2]
        if self._rand_interpolation:
            num = tf.random.uniform([], 0, 2, dtype=tf.int32)
            image = tf.cond(tf.cast(num, dtype=tf.bool),
                            lambda: tf.image.resize_nearest_neighbor(image, re_size, align_corners=True),
                            lambda: tf.image.resize_bilinear(image, re_size, align_corners=True))
        elif self._interpolation.lower() == 'nearest' or self._interpolation.lower() == 'nearest neighbor':
            image = tf.image.resize_nearest_neighbor(image, re_size, align_corners=True)
        elif self._interpolation.lower() == 'bilinear':
            image = tf.image.resize_bilinear(image, re_size, align_corners=True)
        elif self._interpolation.lower() == 'bicubic':
            warnings.warn('Bicubic interpolation is not supported for GPU. Bilinear is used instead.', UserWarning)
            image = tf.image.resize_bilinear(image, re_size, align_corners=True)
        else:
            raise (ValueError, 'Interpolation method of {} is not supported.'.format(self._interpolation))

        image = tf.reshape(image, self.input_size)

        mask = tf.image.resize_nearest_neighbor(mask, re_size, align_corners=True)
        mask = tf.reshape(mask, list(self.input_size[:-1]) + [1])

        return image, mask

    def center_crop(self, x):
        with tf.variable_scope('center_crop'):
            shape_tensor = tf.shape(x)
            h = tf.cast(shape_tensor[1], dtype=tf.float32)
            w = tf.cast(shape_tensor[2], dtype=tf.float32)

            offset_height = tf.cast((h - self.input_size[0])//2, dtype=tf.int32)
            offset_width = tf.cast((w - self.input_size[1])//2, dtype=tf.int32)
            target_height = tf.constant(self.input_size[0], dtype=tf.int32)
            target_width = tf.constant(self.input_size[1], dtype=tf.int32)

            x = tf.slice(x, [0, offset_height, offset_width, 0], [-1, target_height, target_width, -1])

        return x

    def rand_hue(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        max_delta = kwargs.get('rand_hue', 0.2)
        if max_delta > 0.0:
            with tf.variable_scope('rand_hue'):
                if scheduling > 0:
                    max_delta *= self.linear_schedule_multiplier
                elif scheduling < 0:
                    max_delta *= 1.0 - self.linear_schedule_multiplier

                delta = tf.random.uniform([], minval=-max_delta/2, maxval=max_delta/2, dtype=tf.float32)

                x += self.image_mean
                x = tf.image.adjust_hue(x, delta)
                x -= self.image_mean

        return x

    def rand_saturation(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        lower, upper = kwargs.get('rand_saturation', (0.8, 1.25))
        if upper > lower:
            with tf.variable_scope('rand_saturation'):
                if scheduling > 0:
                    lower = tf.math.pow(lower, self.linear_schedule_multiplier)
                    upper = tf.math.pow(upper, self.linear_schedule_multiplier)
                elif scheduling < 0:
                    lower = tf.math.pow(lower, 1.0 - self.linear_schedule_multiplier)
                    upper = tf.math.pow(upper, 1.0 - self.linear_schedule_multiplier)

                base = upper/lower
                randval = tf.random.uniform([], dtype=tf.float32)
                randval = lower*tf.math.pow(base, randval)

                x += self.image_mean
                x = tf.image.adjust_saturation(x, randval)
                x -= self.image_mean

        return x

    def rand_color_balance(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        lower, upper = kwargs.get('rand_color_balance', (1.0, 1.0))
        if upper > lower:
            with tf.variable_scope('random_color_balance'):
                batch_size = tf.shape(x)[0]
                if scheduling > 0:
                    lower = tf.math.pow(lower, self.linear_schedule_multiplier)
                    upper = tf.math.pow(upper, self.linear_schedule_multiplier)
                elif scheduling < 0:
                    lower = tf.math.pow(lower, 1.0 - self.linear_schedule_multiplier)
                    upper = tf.math.pow(upper, 1.0 - self.linear_schedule_multiplier)

                base = upper/lower
                randvals = tf.random.uniform([batch_size, 1, 1, 3], dtype=tf.float32)
                randvals = lower*tf.math.pow(base, randvals)

                image_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

                x -= image_mean
                x = x*randvals
                x += image_mean

        return x

    def rand_equalization(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        prob = kwargs.get('rand_equalization', 0.0)
        if prob > 0.0:
            with tf.variable_scope('random_equalization'):
                batch_size = tf.shape(x)[0]
                if scheduling > 0:
                    prob *= self.linear_schedule_multiplier
                elif scheduling < 0:
                    prob *= 1.0 - self.linear_schedule_multiplier

                normal = tf.cast(tf.greater(tf.random.uniform([batch_size, 1, 1, 1]), prob), dtype=tf.float32)
                maxvals = tf.reduce_max(tf.math.abs(x), axis=[1, 2], keepdims=True)

                image_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

                x -= image_mean
                x = normal*x + (1.0 - normal)*x/maxvals*0.5
                x += image_mean

        return x

    def rand_contrast(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        lower, upper = kwargs.get('rand_contrast', (0.8, 1.25))
        if upper > lower:
            with tf.variable_scope('random_contrast'):
                batch_size = tf.shape(x)[0]
                if scheduling > 0:
                    lower = tf.math.pow(lower, self.linear_schedule_multiplier)
                    upper = tf.math.pow(upper, self.linear_schedule_multiplier)
                elif scheduling < 0:
                    lower = tf.math.pow(lower, 1.0 - self.linear_schedule_multiplier)
                    upper = tf.math.pow(upper, 1.0 - self.linear_schedule_multiplier)

                base = upper/lower
                randvals = tf.random.uniform([batch_size, 1, 1, 1], dtype=tf.float32)
                randvals = lower*tf.math.pow(base, randvals)

                image_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

                x -= image_mean
                x = x*randvals
                x += image_mean

        return x

    def rand_brightness(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        max_delta = kwargs.get('rand_brightness', 0.2)
        if max_delta > 0.0:
            with tf.variable_scope('random_brightness'):
                batch_size = tf.shape(x)[0]
                if scheduling > 0:
                    max_delta *= self.linear_schedule_multiplier
                elif scheduling < 0:
                    max_delta *= 1.0 - self.linear_schedule_multiplier

                randval = tf.random.uniform([batch_size, 1, 1, 1], minval=-max_delta/2, maxval=max_delta/2,
                                            dtype=tf.float32)
                x = x + randval

        return x

    def rand_noise(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        noise_mean = kwargs.get('rand_noise_mean', 0.0)
        noise_stddev = kwargs.get('rand_noise_stddev', 0.0)
        if noise_mean > 0.0 or noise_stddev > 0.0:
            with tf.variable_scope('rand_noise'):
                shape_tensor = tf.shape(x)
                batch_size = shape_tensor[0]

                noise = tf.random.normal(shape_tensor, mean=noise_mean, stddev=noise_stddev, dtype=tf.float32)
                noise_mask = tf.random.uniform([batch_size, 1, 1, 1], dtype=tf.float32)
                if scheduling > 0:
                    noise_mask *= self.linear_schedule_multiplier
                elif scheduling < 0:
                    noise_mask *= 1.0 - self.linear_schedule_multiplier
                x = x + noise_mask*noise

        return x

    def rand_solarization(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        lower, upper = kwargs.get('rand_solarization', (0.0, 1.0))
        if lower > 0.0 or upper < 1.0:
            with tf.variable_scope('rand_solarization'):
                shape_tensor = tf.shape(x)
                batch_size = shape_tensor[0]
                if scheduling > 0:
                    lower = lower*self.linear_schedule_multiplier
                    upper = 1.0 - (1.0 - upper)*self.linear_schedule_multiplier
                elif scheduling < 0:
                    lower = lower*(1.0 - self.linear_schedule_multiplier)
                    upper = 1.0 - (1.0 - upper)*(1.0 - self.linear_schedule_multiplier)

                x += self.image_mean

                thres_lower = tf.random.uniform([batch_size, 1, 1, 1], 0.0, lower, dtype=tf.float32)
                thres_lower = tf.broadcast_to(thres_lower, shape_tensor)
                lower_pixels = tf.less(x, thres_lower)

                thres_upper = tf.random.uniform([batch_size, 1, 1, 1], upper, 1.0, dtype=tf.float32)
                thres_upper = tf.broadcast_to(thres_upper, shape_tensor)
                upper_pixels = tf.greater(x, thres_upper)

                invert = tf.cast(tf.logical_or(lower_pixels, upper_pixels), dtype=tf.float32)

                x = invert*(1.0 - x) + (1.0 - invert)*x
                x -= self.image_mean

        return x

    def rand_posterization(self, x, **kwargs):
        scheduling = kwargs.get('rand_distortion_scheduling', False)
        lower, upper = kwargs.get('rand_posterization', (8, 8))
        if lower < upper or upper < 8:
            with tf.variable_scope('rand_posterization'):
                batch_size = tf.shape(x)[0]
                if scheduling > 0:
                    lower = upper - (upper - lower)*self.linear_schedule_multiplier
                elif scheduling < 0:
                    lower = upper - (upper - lower)*(1.0 - self.linear_schedule_multiplier)

                factors = tf.math.round(tf.random.uniform([batch_size, 1, 1, 1],
                                                          lower - 0.5, upper + 0.5, dtype=tf.float32))
                maxvals = tf.pow(2.0, factors)
                x = tf.math.round(x*maxvals)
                x = x/maxvals

        return x

    def cutmix(self, x, y):
        with tf.variable_scope('cutmix'):
            shape_tensor = tf.shape(x)
            batch_size = shape_tensor[0]
            h = tf.cast(shape_tensor[1], tf.float32)
            w = tf.cast(shape_tensor[2], tf.float32)

            randval = tf.random.uniform([], dtype=tf.float32)
            if self._cutmix_scheduling > 0:
                randval *= self.linear_schedule_multiplier
            elif self._cutmix_scheduling < 0:
                randval *= 1.0 - self.linear_schedule_multiplier
            r_h = tf.random.uniform([], 0, h, dtype=tf.float32)
            r_w = tf.random.uniform([], 0, w, dtype=tf.float32)
            size_h = h*tf.math.sqrt(randval)
            size_w = w*tf.math.sqrt(randval)

            hs = tf.cast(tf.math.round(tf.math.maximum(r_h - size_h/2, 0)), dtype=tf.int32)
            he = tf.cast(tf.math.round(tf.math.minimum(r_h + size_h/2, h)), dtype=tf.int32)
            ws = tf.cast(tf.math.round(tf.math.maximum(r_w - size_w/2, 0)), dtype=tf.int32)
            we = tf.cast(tf.math.round(tf.math.minimum(r_w + size_w/2, w)), dtype=tf.int32)

            m = tf.ones([1, he - hs, we - ws, 1], dtype=tf.float32)
            paddings = [[0, 0],
                        [hs, shape_tensor[1] - he],
                        [ws, shape_tensor[2] - we],
                        [0, 0]]
            m = tf.pad(m, paddings, constant_values=0.0)

            lamb = 1.0 - (tf.cast((he - hs)*(we - ws), dtype=tf.float32))/(h*w)

            idx = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)
            shuffled_x = tf.gather(x, idx, axis=0)
            shuffled_y = tf.gather(y, idx, axis=0)

            x = (1.0 - m)*x + m*shuffled_x
            y = lamb*y + (1.0 - lamb)*shuffled_y

        return x, y

    def weight_variable(self, shape, initializer=tf.initializers.he_normal(),
                        weight_standardization=False, paddings=((0, 0), (0, 0)), name='weights'):
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block is None or self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        with tf.device(self.param_device):
            weights = tf.get_variable(name, shape, dtype=tf.float32,
                                      initializer=initializer,
                                      trainable=False)
            weights_ema = tf.get_variable(name + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                          initializer=weights.initialized_value(), trainable=False)

            if not tf.get_variable_scope().reuse:
                tf.add_to_collection('main_variables', weights)
                tf.add_to_collection('main_variables', weights_ema)
                tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), weights)
                tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), weights_ema)

        with tf.device('/gpu:{}'.format(self._curr_device)), \
             tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
            weights_dev = tf.get_variable(name, shape, dtype=tf.float32,
                                          initializer=initializer, trainable=trainable)

        tf.add_to_collection('weight_variables', weights_dev)
        tf.add_to_collection('block_{}_variables'.format(self._curr_block), weights_dev)
        tf.add_to_collection('block_{}_weight_variables'.format(self._curr_block), weights_dev)
        if trainable:
            tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), weights_dev)

        with tf.variable_scope(self.top_scope):
            self.update_ops.append(self.ema.apply([weights_dev]))
        weights_dev_ema = self.ema.average(weights_dev)
        tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), weights_ema)

        if not tf.get_variable_scope().reuse:
            tf.add_to_collections('var_update_ops', weights.assign(weights_dev))
            tf.add_to_collections('var_update_ops', weights_ema.assign(weights_dev_ema))
        tf.add_to_collections('var_assign_ops', weights_dev.assign(weights))
        tf.add_to_collections('var_assign_ops', weights_dev_ema.assign(weights_ema))

        weights = tf.cond(self.is_train,
                          lambda: weights_dev,
                          lambda: weights_dev_ema)

        if weight_standardization:
            with tf.variable_scope('ws'):
                w_len = len(shape)
                w_idx = list(range(w_len))
                # while len(w_idx) > 2 and shape[w_idx[-1]] == 1:  # Idx correction for depthwise convolution
                #     w_idx = w_idx[:-1]
                mean = tf.math.reduce_mean(weights, axis=w_idx[:-1], keepdims=True)
                weights = weights - mean
                std = tf.math.reduce_std(weights, axis=w_idx[:-1], keepdims=True)
                weights = weights/(std + 1e-5)

        if self.dtype is not tf.float32:
            weights = tf.cast(weights, dtype=self.dtype)

        if paddings[0][0] > 0 or paddings[0][1] > 0 or paddings[1][0] > 0 or paddings[1][1] > 0:
            paddings = list(paddings) + [(0, 0), (0, 0)]
            weights = tf.pad(weights, paddings)

        if self.dropout_weights:
            return tf.nn.dropout(weights, rate=self.dropout_rate_weights)
        else:
            return weights

    def bias_variable(self, shape, initializer=tf.initializers.zeros(), name='biases'):
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block is None or self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        with tf.device(self.param_device):
            biases = tf.get_variable(name, shape, dtype=tf.float32,
                                     initializer=initializer, trainable=False)
            biases_ema = tf.get_variable(name + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                         initializer=biases.initialized_value(), trainable=False)

            if not tf.get_variable_scope().reuse:
                tf.add_to_collection('main_variables', biases)
                tf.add_to_collection('main_variables', biases_ema)
                tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), biases)
                tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), biases_ema)

        with tf.device('/gpu:{}'.format(self._curr_device)), \
             tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
            biases_dev = tf.get_variable(name, shape, dtype=tf.float32,
                                         initializer=initializer, trainable=trainable)

        tf.add_to_collection('bias_variables', biases_dev)
        tf.add_to_collection('block_{}_variables'.format(self._curr_block), biases_dev)
        tf.add_to_collection('block_{}_bias_variables'.format(self._curr_block), biases_dev)
        if trainable:
            tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), biases_dev)

        with tf.variable_scope(self.top_scope):
            self.update_ops.append(self.ema.apply([biases_dev]))
        biases_dev_ema = self.ema.average(biases_dev)
        tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), biases_ema)

        if not tf.get_variable_scope().reuse:
            tf.add_to_collections('var_update_ops', biases.assign(biases_dev))
            tf.add_to_collections('var_update_ops', biases_ema.assign(biases_dev_ema))
        tf.add_to_collections('var_assign_ops', biases_dev.assign(biases))
        tf.add_to_collections('var_assign_ops', biases_dev_ema.assign(biases_ema))

        biases = tf.cond(self.is_train,
                         lambda: biases_dev,
                         lambda: biases_dev_ema)

        if self.dtype is not tf.float32:
            biases = tf.cast(biases, dtype=self.dtype)

        return biases

    def pooling_layer(self, x, kernel, stride, padding='SAME', pooling_type='AVG'):
        if pooling_type.lower() == 'avg':
            return self.avg_pool(x, kernel, stride, padding=padding)
        elif pooling_type.lower() == 'max':
            return self.max_pool(x, kernel, stride, padding=padding)
        else:
            raise ValueError('Pooling type of {} is not supported'.format(pooling_type))

    def max_pool(self, x, side_l, stride, padding='SAME'):
        if not isinstance(side_l, list):
            side_l = [side_l, side_l]
        elif len(side_l) == 1:
            side_l = [side_l[0], side_l[0]]
        if not isinstance(stride, list):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        if self.channel_first:
            ksize = [1, 1, side_l[0], side_l[1]]
            strides = [1, 1, stride[0], stride[1]]
            data_format = 'NCHW'
            _, in_channels, h, w = x.get_shape().as_list()
        else:
            ksize = [1, side_l[0], side_l[1], 1]
            strides = [1, stride[0], stride[1], 1]
            data_format = 'NHWC'
            _, h, w, in_channels = x.get_shape().as_list()

        if padding.lower() == 'same':
            out_size = [np.ceil(float(h)/stride[0]), np.ceil(float(w)/stride[1])]
        else:
            out_size = [np.ceil(float(h - ksize[0] + 1)/stride[0]), np.ceil(float(w - ksize[1] + 1)/stride[1])]

        if self._curr_device == self.gpu_offset:
            self._flops += side_l[0]*side_l[1]*out_size[0]*out_size[1]*in_channels

        return tf.nn.max_pool(x, ksize=ksize, strides=strides, data_format=data_format, padding=padding)

    def avg_pool(self, x, side_l, stride, padding='SAME'):
        if not isinstance(side_l, list):
            side_l = [side_l, side_l]
        elif len(side_l) == 1:
            side_l = [side_l[0], side_l[0]]
        if not isinstance(stride, list):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        if self.channel_first:
            ksize = [1, 1, side_l[0], side_l[1]]
            strides = [1, 1, stride[0], stride[1]]
            data_format = 'NCHW'
            _, in_channels, h, w = x.get_shape().as_list()
        else:
            ksize = [1, side_l[0], side_l[1], 1]
            strides = [1, stride[0], stride[1], 1]
            data_format = 'NHWC'
            _, h, w, in_channels = x.get_shape().as_list()

        if padding.lower() == 'same':
            out_size = [np.ceil(float(h)/stride[0]), np.ceil(float(w)/stride[1])]
        else:
            out_size = [np.ceil(float(h - ksize[0] + 1)/stride[0]), np.ceil(float(w - ksize[1] + 1)/stride[1])]

        if self._curr_device == self.gpu_offset:
            self._flops += side_l[0]*side_l[1]*out_size[0]*out_size[1]*in_channels

        return tf.nn.avg_pool(x, ksize=ksize, strides=strides, data_format=data_format, padding=padding)

    def conv_layer(self, x, kernel, stride, out_channels=None, padding='SAME', biased=True, depthwise=False, scope=None,
                   dilation=(1, 1), ws=False, kernel_paddings=((0, 0), (0, 0)),
                   weight_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.zeros(),
                   verbose=False):
        if not isinstance(kernel, (list, tuple)):
            kernel = [kernel, kernel]
        elif len(kernel) == 1:
            kernel = [kernel[0], kernel[0]]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation, dilation]
        elif len(dilation) == 1:
            dilation = [dilation[0], dilation[0]]

        if self.channel_first:
            _, in_channels, h, w = x.get_shape().as_list()
            conv_strides = [1, 1, stride[0], stride[1]]
            conv_dilations = [1, 1, dilation[0], dilation[1]]
            data_format = 'NCHW'
        else:
            _, h, w, in_channels = x.get_shape().as_list()
            conv_strides = [1, stride[0], stride[1], 1]
            conv_dilations = [1, dilation[0], dilation[1], 1]
            data_format = 'NHWC'

        if padding.lower() == 'same':
            out_size = [np.ceil(float(h)/stride[0]), np.ceil(float(w)/stride[1])]
        else:
            out_size = [np.ceil(float(h - kernel[0] + 1)/stride[0]), np.ceil(float(w - kernel[1] + 1)/stride[1])]

        if out_channels is None:
            out_channels = in_channels

        with tf.variable_scope(scope) if scope is not None else nullcontext():
            if depthwise:
                channel_multiplier = out_channels//in_channels
                if channel_multiplier < 1:
                    channel_multiplier = 1
                weights = self.weight_variable([kernel[0], kernel[1], in_channels, channel_multiplier],
                                               initializer=weight_initializer,
                                               weight_standardization=ws, paddings=kernel_paddings)
                convs = tf.nn.depthwise_conv2d(x, weights, strides=conv_strides, padding=padding,
                                               data_format=data_format, rate=dilation)

                if not tf.get_variable_scope().reuse:
                    self._params += kernel[0]*kernel[1]*in_channels*channel_multiplier
                if self._curr_device == self.gpu_offset:
                    self._flops += out_size[0]*out_size[1]*kernel[0]*kernel[1]*in_channels*channel_multiplier
            else:
                weights = self.weight_variable([kernel[0], kernel[1], in_channels, out_channels],
                                               initializer=weight_initializer,
                                               weight_standardization=ws, paddings=kernel_paddings)
                convs = tf.nn.conv2d(x, weights, strides=conv_strides, padding=padding,
                                     data_format=data_format, dilations=conv_dilations)

                if not tf.get_variable_scope().reuse:
                    self._params += kernel[0]*kernel[1]*in_channels*out_channels
                if self._curr_device == self.gpu_offset:
                    self._flops += out_size[0]*out_size[1]*kernel[0]*kernel[1]*in_channels*out_channels

            if verbose:
                print(tf.get_variable_scope().name + ': [{}, {}, {}], k=({}, {}), s=({}, {}), c={}'
                      .format(h, w, in_channels, kernel[0], kernel[1], stride[0], stride[1], out_channels), end='')
                if depthwise:
                    print(', depthwise.')
                else:
                    print('.')

            if biased:
                biases = self.bias_variable(out_channels, initializer=bias_initializer)

                if not tf.get_variable_scope().reuse:
                    self._params += out_channels
                if self._curr_device == self.gpu_offset:
                    self._flops += out_size[0]*out_size[1]*out_channels

                return tf.nn.bias_add(convs, biases, data_format=data_format)
            else:
                return convs

    def fc_layer(self, x, out_dim, biased=True, scope=None, ws=False,
                 weight_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.zeros(),
                 verbose=False):
        in_dim = int(x.get_shape()[-1])

        with tf.variable_scope(scope) if scope is not None else nullcontext():
            weights = self.weight_variable([in_dim, out_dim], initializer=weight_initializer, weight_standardization=ws)

            if not tf.get_variable_scope().reuse:
                self._params += in_dim*out_dim
            if self._curr_device == self.gpu_offset:
                self._flops += in_dim*out_dim

            if verbose:
                print(tf.get_variable_scope().name + ': [{}, {}]'.format(in_dim, out_dim))

            if biased:
                biases = self.bias_variable(out_dim, initializer=bias_initializer)

                if not tf.get_variable_scope().reuse:
                    self._params += out_dim
                if self._curr_device == self.gpu_offset:
                    self._flops += out_dim

                return tf.matmul(x, weights) + biases
            else:
                return tf.matmul(x, weights)

    def batch_norm(self, x, scale=True, shift=True, zero_scale_init=False, epsilon=1e-3, scope='bn'):
        if self.update_batch_norm is not None:
            update = self.update_batch_norm
        else:
            if self.blocks_to_train is None:
                update = True
            elif self._curr_block is None or self._curr_block in self.blocks_to_train:
                update = True
            else:
                update = False
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block is None or self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        if self.channel_first:
            _, in_channels, h, w = x.get_shape().as_list()
        else:
            _, h, w, in_channels = x.get_shape().as_list()
        momentum = self.batch_norm_decay

        with tf.variable_scope(scope):
            with tf.device(self.param_device):
                mu = tf.get_variable('mu', in_channels, dtype=tf.float32,
                                     initializer=tf.zeros_initializer(), trainable=False)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', mu)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), mu)
                    tf.add_to_collection('norm_statistics', mu)
                    tf.add_to_collection('block_{}_norm_statistics'.format(self._curr_block), mu)
                    with tf.variable_scope(self.top_scope):
                        self.update_ops.append(self.ema.apply([mu]))
                # if self._curr_device == self.gpu_offset:
                #     self._flops += h*w*in_channels
                mu_ema = self.ema.average(mu)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', mu_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), mu_ema)
                    tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), mu_ema)

                sigma = tf.get_variable('sigma', in_channels, dtype=tf.float32,
                                        initializer=tf.ones_initializer(), trainable=False)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', sigma)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), sigma)
                    tf.add_to_collection('norm_statistics', sigma)
                    tf.add_to_collection('block_{}_norm_statistics'.format(self._curr_block), sigma)
                    with tf.variable_scope(self.top_scope):
                        self.update_ops.append(self.ema.apply([sigma]))
                # if self._curr_device == self.gpu_offset:
                #     self._flops += h*w*in_channels
                sigma_ema = self.ema.average(sigma)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', sigma_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), sigma_ema)
                    tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), sigma_ema)

            if scale:
                with tf.device(self.param_device):
                    scale_initializer = tf.zeros_initializer() if zero_scale_init else tf.ones_initializer()
                    gamma = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                            initializer=scale_initializer, trainable=False)
                    gamma_ema = tf.get_variable('gamma' + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                                initializer=gamma.initialized_value(), trainable=False)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', gamma)
                    tf.add_to_collection('main_variables', gamma_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), gamma)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), gamma_ema)

                with tf.device('/gpu:{}'.format(self._curr_device)), \
                     tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
                    gamma_dev = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                                initializer=scale_initializer, trainable=trainable)

                tf.add_to_collection('norm_variables', gamma_dev)
                tf.add_to_collection('block_{}_variables'.format(self._curr_block), gamma_dev)
                tf.add_to_collection('block_{}_norm_variables'.format(self._curr_block), gamma_dev)
                if trainable:
                    tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), gamma_dev)

                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([gamma_dev]))
                gamma_dev_ema = self.ema.average(gamma_dev)
                tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), gamma_ema)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collections('var_update_ops', gamma.assign(gamma_dev))
                    tf.add_to_collections('var_update_ops', gamma_ema.assign(gamma_dev_ema))
                tf.add_to_collections('var_assign_ops', gamma_dev.assign(gamma))
                tf.add_to_collections('var_assign_ops', gamma_dev_ema.assign(gamma_ema))
            else:
                gamma_dev = None
                gamma_dev_ema = None

            if shift:
                with tf.device(self.param_device):
                    beta = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                           initializer=tf.zeros_initializer(), trainable=False)
                    beta_ema = tf.get_variable('beta' + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                               initializer=beta.initialized_value(), trainable=False)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', beta)
                    tf.add_to_collection('main_variables', beta_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), beta)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), beta_ema)

                with tf.device('/gpu:{}'.format(self._curr_device)), \
                     tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
                    beta_dev = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                               initializer=tf.zeros_initializer(), trainable=trainable)

                tf.add_to_collection('norm_variables', beta_dev)
                tf.add_to_collection('block_{}_variables'.format(self._curr_block), beta_dev)
                tf.add_to_collection('block_{}_norm_variables'.format(self._curr_block), beta_dev)
                if trainable:
                    tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), beta_dev)

                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([beta_dev]))
                beta_dev_ema = self.ema.average(beta_dev)
                tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), beta_ema)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collections('var_update_ops', beta.assign(beta_dev))
                    tf.add_to_collections('var_update_ops', beta_ema.assign(beta_dev_ema))
                tf.add_to_collections('var_assign_ops', beta_dev.assign(beta))
                tf.add_to_collections('var_assign_ops', beta_dev_ema.assign(beta_ema))
            else:
                beta_dev = None
                beta_dev_ema = None

            if self._curr_device == self.gpu_offset:
                self._flops += h*w*in_channels

            mean, var = tf.cond(self.is_train,
                                lambda: (mu, sigma),
                                lambda: (mu_ema, sigma_ema))

            beta = tf.cond(self.is_train, lambda: beta_dev, lambda: beta_dev_ema) if beta_dev is not None else None
            gamma = tf.cond(self.is_train, lambda: gamma_dev, lambda: gamma_dev_ema) if gamma_dev is not None else None

            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=tf.float32)
            data_format = 'NCHW' if self.channel_first else 'NHWC'
            if update:
                x, batch_mean, batch_var = tf.cond(self.is_train,
                                                   lambda: tf.nn.fused_batch_norm(x,
                                                                                  gamma,
                                                                                  beta,
                                                                                  epsilon=epsilon,
                                                                                  data_format=data_format,
                                                                                  is_training=True),
                                                   lambda: tf.nn.fused_batch_norm(x,
                                                                                  gamma,
                                                                                  beta,
                                                                                  mean=mean,
                                                                                  variance=var,
                                                                                  epsilon=epsilon,
                                                                                  data_format=data_format,
                                                                                  is_training=False)
                                                   )
                update_rate = 1.0 - momentum
                if self._curr_device == self.gpu_offset:
                    update_mu = momentum*mu + update_rate*batch_mean
                    update_sigma = momentum*sigma + update_rate*batch_var
                else:  # Chained variable updates
                    dep_ops = tf.get_collection('dev_{}_update_ops'.format(self._curr_device - 1))
                    updated_mu, updated_sigma = dep_ops[self._curr_dependent_op:self._curr_dependent_op + 2]
                    update_mu = momentum*updated_mu + update_rate*batch_mean
                    update_sigma = momentum*updated_sigma + update_rate*batch_var
                tf.add_to_collection('dev_{}_update_ops'.format(self._curr_device), update_mu)
                tf.add_to_collection('dev_{}_update_ops'.format(self._curr_device), update_sigma)
                self._curr_dependent_op += 2
                if self._curr_device == self.gpu_offset + self.num_gpus - 1:
                    update_mu = mu.assign(update_mu)
                    update_sigma = sigma.assign(update_sigma)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)
            else:
                x, _, _ = tf.nn.fused_batch_norm(x,
                                                 gamma,
                                                 beta,
                                                 mean=mean,
                                                 variance=var,
                                                 epsilon=epsilon,
                                                 data_format=data_format,
                                                 is_training=False)
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=self.dtype)

        return x

    def group_norm(self, x, num_groups=8, scale=True, shift=True, zero_scale_init=False, epsilon=1e-3, scope='gn'):
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block is None or self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(scope):
            batch_size = tf.shape(x)[0]
            in_shape = x.get_shape().as_list()
            if len(in_shape) > 2:
                if self.channel_first:
                    _, in_channels, h, w = in_shape
                    x_shape = [batch_size, in_channels//num_groups, num_groups, h, w]
                    axis = [1, 3, 4]
                    var_shape = [1, in_channels, 1, 1]
                else:
                    _, h, w, in_channels = in_shape
                    x_shape = [batch_size, h, w, in_channels//num_groups, num_groups]
                    axis = [1, 2, 3]
                    var_shape = [1, 1, 1, in_channels]
            else:
                in_channels = in_shape[1]
                h, w = 1, 1
                x_shape = [batch_size, num_groups, in_channels//num_groups]
                axis = [2]
                var_shape = [1, in_channels]

            if scale:
                with tf.device(self.param_device):
                    scale_initializer = tf.zeros_initializer() if zero_scale_init else tf.ones_initializer()
                    gamma = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                            initializer=scale_initializer, trainable=False)
                    gamma_ema = tf.get_variable('gamma' + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                                initializer=gamma.initialized_value(), trainable=False)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', gamma)
                    tf.add_to_collection('main_variables', gamma_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), gamma)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), gamma_ema)

                with tf.device('/gpu:{}'.format(self._curr_device)), \
                     tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
                    gamma_dev = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                                initializer=scale_initializer, trainable=trainable)

                tf.add_to_collection('norm_variables', gamma_dev)
                tf.add_to_collection('block_{}_variables'.format(self._curr_block), gamma_dev)
                tf.add_to_collection('block_{}_norm_variables'.format(self._curr_block), gamma_dev)
                if trainable:
                    tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), gamma_dev)

                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([gamma_dev]))
                gamma_dev_ema = self.ema.average(gamma_dev)
                tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), gamma_ema)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collections('var_update_ops', gamma.assign(gamma_dev))
                    tf.add_to_collections('var_update_ops', gamma_ema.assign(gamma_dev_ema))
                tf.add_to_collections('var_assign_ops', gamma_dev.assign(gamma))
                tf.add_to_collections('var_assign_ops', gamma_dev_ema.assign(gamma_ema))
            else:
                gamma_dev = None
                gamma_dev_ema = None

            if shift:
                with tf.device(self.param_device):
                    beta = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                           initializer=tf.zeros_initializer(), trainable=False)
                    beta_ema = tf.get_variable('beta' + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                               initializer=beta.initialized_value(), trainable=False)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', beta)
                    tf.add_to_collection('main_variables', beta_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), beta)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), beta_ema)

                with tf.device('/gpu:{}'.format(self._curr_device)), \
                     tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
                    beta_dev = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                               initializer=tf.zeros_initializer(), trainable=trainable)

                tf.add_to_collection('norm_variables', beta_dev)
                tf.add_to_collection('block_{}_variables'.format(self._curr_block), beta_dev)
                tf.add_to_collection('block_{}_norm_variables'.format(self._curr_block), beta_dev)
                if trainable:
                    tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), beta_dev)

                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([beta_dev]))
                beta_dev_ema = self.ema.average(beta_dev)
                tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), beta_ema)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collections('var_update_ops', beta.assign(beta_dev))
                    tf.add_to_collections('var_update_ops', beta_ema.assign(beta_dev_ema))
                tf.add_to_collections('var_assign_ops', beta_dev.assign(beta))
                tf.add_to_collections('var_assign_ops', beta_dev_ema.assign(beta_ema))
            else:
                beta_dev = None
                beta_dev_ema = None

                if self._curr_device == self.gpu_offset:
                    self._flops += h*w*in_channels

            beta = tf.cond(self.is_train, lambda: beta_dev, lambda: beta_dev_ema) if beta_dev is not None else None
            gamma = tf.cond(self.is_train, lambda: gamma_dev, lambda: gamma_dev_ema) if gamma_dev is not None else None
            beta = tf.reshape(beta, var_shape) if beta is not None else None
            gamma = tf.reshape(gamma, var_shape) if gamma is not None else None

            x = tf.reshape(x, shape=x_shape)
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=tf.float32)
            mean, var = tf.nn.moments(x, axes=axis, keepdims=True)
            x = (x - mean)/tf.math.sqrt(var + epsilon)
            x = tf.reshape(x, shape=[batch_size] + in_shape[1:])
            x = x*gamma if gamma is not None else x
            x = x + beta if beta is not None else x
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=self.dtype)

        return x

    def group_renorm(self, x, num_groups=8, scale=True, shift=True, zero_scale_init=False, epsilon=1e-3, scope='gn'):
        if self.update_batch_norm is not None:  # Experimental renormalization
            update = self.update_batch_norm
        else:
            if self.blocks_to_train is None:
                update = True
            elif self._curr_block is None or self._curr_block in self.blocks_to_train:
                update = True
            else:
                update = False
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block is None or self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        momentum = self.batch_norm_decay

        with tf.variable_scope(scope):
            batch_size = tf.shape(x)[0]
            in_shape = x.get_shape().as_list()
            if len(in_shape) > 2:
                if self.channel_first:
                    _, in_channels, h, w = in_shape
                    x_shape = [batch_size, in_channels//num_groups, num_groups, h, w]
                    axis = [1, 3, 4]
                    var_shape = [1, in_channels, 1, 1]
                else:
                    _, h, w, in_channels = in_shape
                    x_shape = [batch_size, h, w, in_channels//num_groups, num_groups]
                    axis = [1, 2, 3]
                    var_shape = [1, 1, 1, in_channels]
            else:
                in_channels = in_shape[1]
                h, w = 1, 1
                x_shape = [batch_size, num_groups, in_channels//num_groups]
                axis = [2]
                var_shape = [1, in_channels]

            with tf.device(self.param_device):
                mu = tf.get_variable('mu', in_channels, dtype=tf.float32,
                                     initializer=tf.zeros_initializer(), trainable=False)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', mu)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), mu)
                    tf.add_to_collection('norm_statistics', mu)
                    tf.add_to_collection('block_{}_norm_statistics'.format(self._curr_block), mu)
                    with tf.variable_scope(self.top_scope):
                        self.update_ops.append(self.ema.apply([mu]))
                # if self._curr_device == self.gpu_offset:
                #     self._flops += h*w*in_channels
                mu_ema = self.ema.average(mu)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', mu_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), mu_ema)
                    tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), mu_ema)

                sigma = tf.get_variable('sigma', in_channels, dtype=tf.float32,
                                        initializer=tf.ones_initializer(), trainable=False)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', sigma)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), sigma)
                    tf.add_to_collection('norm_statistics', sigma)
                    tf.add_to_collection('block_{}_norm_statistics'.format(self._curr_block), sigma)
                    with tf.variable_scope(self.top_scope):
                        self.update_ops.append(self.ema.apply([sigma]))
                # if self._curr_device == self.gpu_offset:
                #     self._flops += h*w*in_channels
                sigma_ema = self.ema.average(sigma)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', sigma_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), sigma_ema)
                    tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), sigma_ema)

            if scale:
                with tf.device(self.param_device):
                    scale_initializer = tf.zeros_initializer() if zero_scale_init else tf.ones_initializer()
                    gamma = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                            initializer=scale_initializer, trainable=False)
                    gamma_ema = tf.get_variable('gamma' + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                                initializer=gamma.initialized_value(), trainable=False)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', gamma)
                    tf.add_to_collection('main_variables', gamma_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), gamma)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), gamma_ema)

                with tf.device('/gpu:{}'.format(self._curr_device)), \
                     tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
                    gamma_dev = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                                initializer=scale_initializer, trainable=trainable)

                tf.add_to_collection('norm_variables', gamma_dev)
                tf.add_to_collection('block_{}_variables'.format(self._curr_block), gamma_dev)
                tf.add_to_collection('block_{}_norm_variables'.format(self._curr_block), gamma_dev)
                if trainable:
                    tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), gamma_dev)

                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([gamma_dev]))
                gamma_dev_ema = self.ema.average(gamma_dev)
                tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), gamma_ema)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collections('var_update_ops', gamma.assign(gamma_dev))
                    tf.add_to_collections('var_update_ops', gamma_ema.assign(gamma_dev_ema))
                tf.add_to_collections('var_assign_ops', gamma_dev.assign(gamma))
                tf.add_to_collections('var_assign_ops', gamma_dev_ema.assign(gamma_ema))
            else:
                gamma_dev = None
                gamma_dev_ema = None

            if shift:
                with tf.device(self.param_device):
                    beta = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                           initializer=tf.zeros_initializer(), trainable=False)
                    beta_ema = tf.get_variable('beta' + '/ExponentialMovingAverage', None, dtype=tf.float32,
                                               initializer=beta.initialized_value(), trainable=False)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('main_variables', beta)
                    tf.add_to_collection('main_variables', beta_ema)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), beta)
                    tf.add_to_collection('block_{}_main_variables'.format(self._curr_block), beta_ema)

                with tf.device('/gpu:{}'.format(self._curr_device)), \
                     tf.variable_scope('dev_{}'.format(self._curr_device), reuse=tf.AUTO_REUSE):
                    beta_dev = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                               initializer=tf.zeros_initializer(), trainable=trainable)

                tf.add_to_collection('norm_variables', beta_dev)
                tf.add_to_collection('block_{}_variables'.format(self._curr_block), beta_dev)
                tf.add_to_collection('block_{}_norm_variables'.format(self._curr_block), beta_dev)
                if trainable:
                    tf.add_to_collection('dev_{}_trainable_variables'.format(self._curr_device), beta_dev)

                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([beta_dev]))
                beta_dev_ema = self.ema.average(beta_dev)
                tf.add_to_collection('block_{}_ema_variables'.format(self._curr_block), beta_ema)

                if not tf.get_variable_scope().reuse:
                    tf.add_to_collections('var_update_ops', beta.assign(beta_dev))
                    tf.add_to_collections('var_update_ops', beta_ema.assign(beta_dev_ema))
                tf.add_to_collections('var_assign_ops', beta_dev.assign(beta))
                tf.add_to_collections('var_assign_ops', beta_dev_ema.assign(beta_ema))
            else:
                beta_dev = None
                beta_dev_ema = None

            if self._curr_device == self.gpu_offset:
                self._flops += h*w*in_channels

            moving_mean, moving_var = tf.cond(self.is_train,
                                              lambda: (mu, sigma),
                                              lambda: (mu_ema, sigma_ema))

            beta = tf.cond(self.is_train, lambda: beta_dev, lambda: beta_dev_ema) if beta_dev is not None else None
            gamma = tf.cond(self.is_train, lambda: gamma_dev, lambda: gamma_dev_ema) if gamma_dev is not None else None
            beta = tf.reshape(beta, var_shape) if beta is not None else None
            gamma = tf.reshape(gamma, var_shape) if gamma is not None else None

            x = tf.reshape(x, shape=x_shape)
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=tf.float32)
            mean, var = tf.nn.moments(x, axes=axis, keepdims=True)
            x = (x - mean)/tf.math.sqrt(var + epsilon)
            x = tf.reshape(x, shape=[batch_size] + in_shape[1:])

            if update:
                axis = [0, 2, 3] if self.channel_first else [0, 1, 2]
                batch_mean, batch_var = tf.nn.moments(x, axes=axis)

                update_rate = 1.0 - momentum
                if self._curr_device == self.gpu_offset:
                    update_mu = momentum*mu + update_rate*batch_mean
                    update_sigma = momentum*sigma + update_rate*batch_var
                else:  # Chained variable updates
                    dep_ops = tf.get_collection('dev_{}_update_ops'.format(self._curr_device - 1))
                    updated_mu, updated_sigma = dep_ops[self._curr_dependent_op:self._curr_dependent_op + 2]
                    update_mu = momentum*updated_mu + update_rate*batch_mean
                    update_sigma = momentum*updated_sigma + update_rate*batch_var
                tf.add_to_collection('dev_{}_update_ops'.format(self._curr_device), update_mu)
                tf.add_to_collection('dev_{}_update_ops'.format(self._curr_device), update_sigma)
                self._curr_dependent_op += 2
                if self._curr_device == self.gpu_offset + self.num_gpus - 1:
                    update_mu = mu.assign(update_mu)
                    update_sigma = sigma.assign(update_sigma)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

            moving_std = tf.math.sqrt(moving_var + epsilon)
            moving_mean = tf.reshape(moving_mean, shape=var_shape)
            moving_std = tf.reshape(moving_std, shape=var_shape)

            gamma_hat = gamma/moving_std if gamma is not None else 1/moving_std
            beta_hat = beta - gamma_hat*moving_mean if beta is not None else -gamma_hat*moving_mean
            x = x*gamma_hat + beta_hat
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=self.dtype)

        return x

    def upsampling_2d_layer(self, x, scale=2, out_shape=None, align_corners=False,
                            upsampling_method='bilinear', name='upsampling'):
        with tf.variable_scope(name):
            if self.channel_first:
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=tf.float32)
            in_shape = x.get_shape()
            if out_shape is None:
                out_shape = [in_shape[1]*scale, in_shape[2]*scale]
            if upsampling_method.lower() == 'nearest' or upsampling_method.lower() == 'nearest_neighbor':
                x = tf.image.resize_nearest_neighbor(x, out_shape, align_corners=align_corners,
                                                     half_pixel_centers=not align_corners, name=name)
            elif upsampling_method.lower() == 'bilinear':
                x = tf.image.resize_bilinear(x, out_shape, align_corners=align_corners,
                                             half_pixel_centers=not align_corners, name=name)
            else:
                raise(ValueError, 'Upsampling method of {} is not supported'.format(upsampling_method))

            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=self.dtype)
            if self.channel_first:
                x = tf.transpose(x, perm=[0, 3, 1, 2])

        return x

    def transposed_conv_layer(self, x, kernel, stride, out_channels, padding='SAME', biased=True, output_shape=None,
                              dilation=(1, 1), scope=None, weight_initializer=tf.initializers.he_normal(),
                              bias_initializer=tf.initializers.zeros(), ws=False):
        if not isinstance(kernel, (list, tuple)):
            kernel = [kernel, kernel]
        elif len(kernel) == 1:
            kernel = [kernel[0], kernel[0]]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation, dilation]
        elif len(dilation) == 1:
            dilation = [dilation[0], dilation[0]]

        batch_size = tf.shape(x)[0]
        if self.channel_first:
            _, in_channels, h, w = x.get_shape().as_list()
            conv_strides = [1, 1, stride[0], stride[1]]
            conv_dilations = [1, 1, dilation[0], dilation[1]]
            data_format = 'NCHW'
            if output_shape is None:
                if padding.lower() == 'valid':
                    output_shape = [batch_size, out_channels, h*stride[0] - kernel[0] + 1, w*stride[1] - kernel[1] + 1]
                else:
                    output_shape = [batch_size, out_channels, h*stride[0], w*stride[1]]
            out_size = output_shape[2:4]
        else:
            _, h, w, in_channels = x.get_shape().as_list()
            conv_strides = [1, stride[0], stride[1], 1]
            conv_dilations = [1, dilation[0], dilation[1], 1]
            data_format = 'NHWC'
            if output_shape is None:
                if padding.lower() == 'valid':
                    output_shape = [batch_size, h*stride[0] - kernel[0] + 1, w*stride[1] - kernel[1] + 1, out_channels]
                else:
                    output_shape = [batch_size, h*stride[0], w*stride[1], out_channels]
            out_size = output_shape[1:3]

        with tf.variable_scope(scope) if scope is not None else nullcontext():
            weights = self.weight_variable([kernel[0], kernel[1], in_channels, out_channels],
                                           initializer=weight_initializer, weight_standardization=ws)
            weights = tf.transpose(weights, perm=[0, 1, 3, 2])
            convs = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape, strides=conv_strides,
                                           padding=padding, data_format=data_format, dilations=conv_dilations)

            if not tf.get_variable_scope().reuse:
                self._params += kernel[0]*kernel[1]*in_channels*out_channels
            if self._curr_device == self.gpu_offset:
                self._flops += out_size[0]*out_size[1]*kernel[0]*kernel[1]*in_channels*out_channels

            if biased:
                biases = self.bias_variable(out_channels, initializer=bias_initializer)

                if not tf.get_variable_scope().reuse:
                    self._params += out_channels
                if self._curr_device == self.gpu_offset:
                    self._flops += out_size[0]*out_size[1]*out_channels

                return tf.nn.bias_add(convs, biases, data_format=data_format)
            else:
                return convs

    def stochastic_depth(self, x, skip, drop_rate=0.0, name='drop'):
        if drop_rate > 0.0:
            with tf.variable_scope(name):
                batch_size = tf.shape(x)[0]
                drop_rate = tf.cond(self.is_train, lambda: drop_rate, lambda: 0.0)

                s = tf.math.greater_equal(tf.random.uniform([batch_size, 1, 1, 1], dtype=tf.float32), drop_rate)
                survived = tf.cast(tf.cast(s, dtype=tf.float32)/(1.0 - drop_rate), dtype=self.dtype)

                x = x*survived + skip
        else:
            x = x + skip

        return x

    def activation(self, x, activation_type='relu'):
        if activation_type.lower() == 'relu':
            return self.relu(x, name=activation_type)
        elif activation_type.lower() == 'swish':
            return self.swish(x, name=activation_type)
        elif activation_type.lower() == 'sigmoid':
            return self.sigmoid(x, name=activation_type)
        else:
            raise ValueError('Activation type of {} is not supported'.format(activation_type))

    def relu(self, x, name='relu'):
        return tf.nn.relu(x, name=name)

    def swish(self, x, name='swish'):
        return tf.nn.swish(x, name=name)

    def sigmoid(self, x, name=None):
        return tf.nn.sigmoid(x, name=name)

    def grad_cam(self, logits, conv_layer, y=None):
        eps = 1e-4
        with tf.variable_scope('grad_cam'):
            if y is None:
                axis = 1 if self.channel_first else -1
                logits_mask = tf.stop_gradient(logits//tf.reduce_max(logits, axis=axis, keepdims=True))
            else:
                logits_mask = y
            logits = logits_mask*logits
            axis = [2, 3] if self.channel_first else [1, 2]
            channel_weights = tf.reduce_sum(tf.gradients(logits, conv_layer)[0], axis=axis, keepdims=True)
            axis = 1 if self.channel_first else -1
            gcam = tf.nn.relu(tf.reduce_sum(channel_weights*conv_layer, axis=axis, keepdims=True))
            if self.dtype is not tf.float32:
                gcam = tf.cast(gcam, dtype=tf.float32)
            if self.channel_first:
                gcam = tf.transpose(gcam, perm=[0, 2, 3, 1])
            gcam = tf.image.resize_bilinear(gcam, self.input_size[0:2], half_pixel_centers=True)
            gcam = gcam/(tf.reduce_max(gcam, axis=[1, 2, 3], keepdims=True) + eps)

            return gcam
