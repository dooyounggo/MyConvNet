"""
Build convolutional neural networks using TensorFlow low-level APIs.

==================== Reference Pages ====================
Basic code structure: http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html
Basic code GitHub: https://github.com/sualab/asirra-dogs-cats-classification
ResNet: https://arxiv.org/abs/1512.03385
ResNet-18 example code: https://github.com/dalgu90/resnet-18-tensorflow
ResNet with Identity Mappings: https://arxiv.org/abs/1603.05027
Squeeze-and-Excitation: https://arxiv.org/abs/1709.01507
Convolutional Block Attention Module: https://arxiv.org/abs/1807.06521
ResNet-D: https://arxiv.org/abs/1812.01187
EfficientNet: https://arxiv.org/abs/1905.11946
Stochastic Depth: https://arxiv.org/abs/1603.09382
Monte Carlo Dropout: https://arxiv.org/abs/1506.02142
Bag of Tricks: https://arxiv.org/abs/1812.01187
CutMix: https://arxiv.org/abs/1905.04899
"""

import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np


class ConvNet(object):
    def __init__(self, input_shape, num_classes, loss_weights=None, **kwargs):
        graph = tf.get_default_graph()
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = kwargs.get('num_parallel_calls', 0)
        config.inter_op_parallelism_threads = 0
        config.gpu_options.force_gpu_compatible = True
        config.allow_soft_placement = False
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7
        self.session = tf.Session(graph=graph, config=config)  # TF main session
        self.top_scope = tf.get_variable_scope()

        self._input_size = input_shape  # Size of the network input (i.e., the first convolution layer).
        self._image_size = kwargs.get('image_size', None)
        self._num_classes = num_classes
        self._loss_weights = loss_weights  # Weight values for the softmax losses of each class

        self._dtype = kwargs.get('data_type', tf.float32)
        self._channel_first = kwargs.get('channel_first', False)
        self._num_gpus = kwargs.get('num_gpus', 1)

        self._padded_size = np.round(np.array(self.input_size[0:2])*(1.0 + kwargs.get('zero_pad_ratio', 0.0)))
        self.pad_value = kwargs.get('pad_value', 0.5)

        self._dropout_weights = kwargs.get('dropout_weights', False)
        self._dropout_logits = kwargs.get('dropout_logits', False)

        self._blocks_to_train = kwargs.get('blocks_to_train', None)
        self._train_batch_norm = kwargs.get('train_batch_norm', True)
        self._batch_norm_decay = kwargs.get('batch_norm_decay', 0.999)

        self.debug_value = 0.0
        self.debug_images_0 = np.zeros([4, 8, 8, 3], dtype=np.float32)
        self.debug_images_1 = np.zeros([4, 8, 8, 3], dtype=np.float32)

        self.handles = []
        self.Xs = []
        self.Ys = []
        self.preds = []
        self.losses = []
        self.gcams = []
        self.bytes_in_use = []

        self._flops = 0
        self._params = 0

        self.backbone_only = False
        self.dicts = []
        self._update_ops = []
        self.ema = tf.train.ExponentialMovingAverage(decay=kwargs.get('moving_average_decay', 0.999))
        self._init_params()
        self._init_model(**kwargs)

        print('\nNumber of GPUs : {}'.format(self._num_gpus))
        print('Total number of units: {}'.format(self._num_blocks))
        print('\n# FLOPs : {:-15,}\n# Params: {:-15,}\n'.format(int(self._flops), int(self._params)))

    @property
    def input_size(self):
        return self._input_size

    @property
    def image_size(self):
        return self._image_size

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
    def num_gpus(self):
        return self._num_gpus

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def dropout_weights(self):
        return self._dropout_weights

    @property
    def dropout_logits(self):
        return self._dropout_logits

    @property
    def blocks_to_train(self):
        return self._blocks_to_train

    @property
    def train_batch_norm(self):
        return self._train_batch_norm

    @property
    def batch_norm_decay(self):
        return self._batch_norm_decay

    @property
    def update_ops(self):
        return self._update_ops

    def _init_model(self, **kwargs):
        with tf.device('/cpu:0'):
            with tf.variable_scope('conditions'):
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.monte_carlo = tf.placeholder(tf.bool, name='monte_carlo')
                self.augmentation = tf.placeholder(tf.bool, name='augmentation')
                self.dropout_rate = tf.cond(tf.math.logical_or(self.is_train, self.monte_carlo),
                                            lambda: tf.constant(kwargs.get('dropout_rate', 0.0), dtype=self.dtype),
                                            lambda: tf.constant(0.0, dtype=self.dtype),
                                            name='dropout_rate')
                if self.dropout_weights:
                    self.dropout_rate_weights = self.dropout_rate
                else:
                    self.dropout_rate_weights = tf.constant(0.0, dtype=self.dtype)
                if self.dropout_logits:
                    self.dropout_rate_logits = self.dropout_rate
                else:
                    self.dropout_rate_logits = tf.constant(0.0, dtype=self.dtype)
                if kwargs.get('zero_center', True):
                    self.image_mean = kwargs.get('image_mean', 0.5)
                else:
                    self.image_mean = 0.0

        self.X_in = []
        self.Y_in = []
        if self.image_size is None:
            output_shapes = ([None, None, None, self.input_size[-1]],
                             [None])
        else:
            output_shapes = ([None, self.image_size[0], self.image_size[1], self.input_size[-1]],
                             [None])
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self._num_gpus):
                self._curr_block = 0
                self._num_blocks = 0
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

                        self.X = self.zero_pad(self.X)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        self.X = tf.cond(self.augmentation,
                                         lambda: self.augment_images(self.X, **kwargs),
                                         lambda: self.center_crop(self.X),
                                         name='augmentation')
                        if kwargs.get('cutmix', False):
                            self.X, self.Y = tf.cond(self.is_train,
                                                     lambda: self.cutmix(self.X, self.Y),
                                                     lambda: (self.X, self.Y),
                                                     name='cutmix')
                        self.Xs.append(self.X)
                        self.Ys.append(self.Y)

                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])

                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.X = tf.cast(self.X, dtype=self.dtype)
                        self.d = self._build_model(**kwargs)
                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.d['logits'] = tf.cast(self.d['logits'], dtype=tf.float32)
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        tf.get_variable_scope().reuse_variables()

                        self.dicts.append(self.d)
                        self.gcams.append(self.grad_cam(self.d['logits'],
                                                        self.d['block_{}'.format(self.num_blocks - 1)],
                                                        y=None))

                        self.logits = self.d['logits']
                        self.preds.append(self.d['pred'])
                        self.losses.append(self._build_loss(**kwargs))

                        self.bytes_in_use.append(tf.contrib.memory_stats.BytesInUse())

        with tf.device('/cpu:0'):
            with tf.variable_scope('calc'):
                self.X_all = tf.concat(self.Xs, axis=0, name='x') + self.image_mean
                self.Y_all = tf.concat(self.Ys, axis=0, name='y_true')
                self.pred = tf.concat(self.preds, axis=0, name='y_pred')
                self.loss = tf.reduce_mean(self.losses, name='mean_loss')
                self.gcam = tf.concat(self.gcams, axis=0, name='grad_cam')

                self.input_images = tf.concat(self.X_in, axis=0, name='x_in')
                self.debug_images_0 = tf.clip_by_value(self.gcam/2 + self.X_all + 0.5, 0, 1)
                self.debug_images_1 = tf.clip_by_value(self.gcam*(self.X_all + 0.5), 0, 1)

    @abstractmethod
    def _init_params(self):
        """
        Parameter initialization.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    def _build_loss(self, **kwargs):
        l1_factor = kwargs.get('l1_reg', 1e-8)
        l2_factor = kwargs.get('l2_reg', 1e-4)
        ls_factor = kwargs.get('label_smoothing', 0.0)
        variables = tf.get_collection('weight_variables')
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
            if self.channel_first:
                axis = 1
                while len(w.get_shape()) < len(self.Y.get_shape()):
                    w = tf.expand_dims(w, axis=-1)
            else:
                axis = -1
                while len(w.get_shape()) < len(self.Y.get_shape()):
                    w = tf.expand_dims(w, axis=1)
            batch_weights = tf.reduce_sum(self.Y*w, axis=axis)

            with tf.variable_scope('l1_loss'):
                if l1_factor > 0.0:
                    l1_factor = tf.constant(l1_factor, dtype=tf.float32, name='L1_factor')
                    l1_reg_loss = l1_factor*tf.add_n([tf.reduce_sum(tf.abs(var)) for var in variables])
                else:
                    l1_reg_loss = tf.constant(0.0, dtype=tf.float32, name='0')
            with tf.variable_scope('l2_loss'):
                if l2_factor > 0.0:
                    l2_factor = tf.constant(l2_factor, dtype=tf.float32, name='L2_factor')
                    l2_reg_loss = l2_factor*tf.math.accumulate_n([tf.nn.l2_loss(var) for var in variables])
                else:
                    l2_reg_loss = tf.constant(0.0, dtype=tf.float32, name='0')

            with tf.variable_scope('label_smoothing'):
                ls_factor = tf.constant(ls_factor, dtype=tf.float32, name='label_smoothing_factor')
                labels = self.Y*(1.0 - ls_factor) + ls_factor/self.num_classes

            with tf.variable_scope('valid_mask'):
                sumval = tf.reduce_sum(self.Y, axis=axis)
                valid_g = tf.greater(sumval, 1.0 - valid_eps)
                valid_l = tf.less(sumval, 1.0 + valid_eps)
                valid = tf.cast(tf.logical_and(valid_g, valid_l), dtype=tf.float32)

            softmax_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.logits, axis=axis)
            softmax_loss = tf.reduce_mean(batch_weights*valid*softmax_losses)

            loss = softmax_loss + l1_reg_loss + l2_reg_loss

        return loss

    def predict(self, dataset, verbose=False, return_images=True, **kwargs):
        batch_size = kwargs.get('batch_size', 32)
        augment_pred = kwargs.get('augment_pred', False)

        pred_size = dataset.num_examples
        num_steps = np.ceil(pred_size/batch_size).astype(int)
        monte_carlo = kwargs.get('monte_carlo', False)

        dataset.initialize(self.session)
        handles = dataset.get_string_handles(self.session)

        if verbose:
            print('Running prediction loop...')

        feed_dict = {self.is_train: False,
                     self.monte_carlo: monte_carlo,
                     self.augmentation: augment_pred}
        for h_t, h in zip(self.handles, handles):
            feed_dict.update({h_t: h})

        if return_images:
            _X = np.zeros([pred_size] + list(self.input_size), dtype=float)
        else:
            _X = np.zeros([pred_size] + [4, 4, 3], dtype=float)  # Dummy images
        _Y_true = np.zeros([pred_size] + self.pred.get_shape().as_list()[1:], dtype=float)
        _Y_pred = np.zeros([pred_size] + self.pred.get_shape().as_list()[1:], dtype=float)
        _loss_pred = np.zeros(pred_size, dtype=float)
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
        batch_size = kwargs.get('batch_size', 32)
        augment_pred = kwargs.get('augment_pred', False)

        pred_size = dataset.num_examples
        num_steps = np.ceil(pred_size / batch_size).astype(int)
        monte_carlo = kwargs.get('monte_carlo', False)

        dataset.initialize(self.session)
        handles = dataset.get_string_handles(self.session)

        feed_dict = {self.is_train: False,
                     self.monte_carlo: monte_carlo,
                     self.augmentation: augment_pred}
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

    def zero_pad(self, x):
        with tf.variable_scope('zero_pad'):
            x = tf.map_fn(self.zero_pad_image, x, parallel_iterations=32, back_prop=False)

        return x

    def zero_pad_image(self, x):
        shape_tensor = tf.cast(tf.shape(x), dtype=tf.float32)
        h = shape_tensor[0]
        w = shape_tensor[1]
        pad_h = tf.maximum(self._padded_size[0] - h, 0.0)
        pad_w = tf.maximum(self._padded_size[1] - w, 0.0)
        paddings = [[tf.cast(tf.floor(pad_h/2), dtype=tf.int32), tf.cast(tf.ceil(pad_h/2), dtype=tf.int32)],
                    [tf.cast(tf.floor(pad_w/2), dtype=tf.int32), tf.cast(tf.ceil(pad_w/2), dtype=tf.int32)],
                    [0, 0]]
        x = tf.pad(x, paddings, constant_values=self.pad_value)

        return x

    def augment_images(self, x, mask=None, **kwargs):
        rand_affine = kwargs.get('rand_affine', True)
        rand_crop = kwargs.get('rand_crop', True)
        rand_distortion = kwargs.get('rand_distortion', True)

        if rand_affine:
            x, mask = self.affine_augment(x, mask, **kwargs)

        if rand_crop:
            x, mask = self.rand_crop(x, mask, **kwargs)
        else:
            x, mask = (self.center_crop(x), self.center_crop(mask))

        if rand_distortion:
            x = self.rand_hue(x, **kwargs)
            x = self.rand_saturation(x, **kwargs)
            x = self.rand_color_balance(x, **kwargs)
            x = self.rand_equalization(x, **kwargs)
            x = self.rand_contrast(x, **kwargs)
            x = self.rand_brightness(x, **kwargs)
            x = self.rand_noise(x, **kwargs)
            x = tf.clip_by_value(x, -0.5, 0.5)
            x = self.rand_solarization(x, **kwargs)
            x = self.rand_posterization(x, **kwargs)

        if mask is None:
            return x
        else:
            return x, mask

    def affine_augment(self, x, mask=None, **kwargs):  # Scale, ratio, translation, rotation, shear, and reflection
        with tf.variable_scope('affine_augment'):
            shape_tensor = tf.shape(x)
            batch_size = shape_tensor[0]
            H = tf.cast(shape_tensor[1], dtype=tf.float32)
            W = tf.cast(shape_tensor[2], dtype=tf.float32)

            lower, upper = kwargs.get('rand_scale', (1.0, 1.0))
            base = float(upper/lower)
            randvals = tf.random.uniform([batch_size, 1], dtype=tf.float32)
            rand_scale = lower*tf.math.pow(base, randvals)

            lower, upper = kwargs.get('rand_ratio', (1.0, 1.0))
            base = float(upper/lower)
            randvals = tf.random.uniform([batch_size, 1], dtype=tf.float32)
            rand_ratio = lower*tf.math.pow(base, randvals)

            rand_x_scale = rand_scale*tf.math.sqrt(rand_ratio)
            rand_y_scale = rand_scale/tf.math.sqrt(rand_ratio)

            rand_rotation = (tf.random.uniform([batch_size, 1]) - 0.5)*kwargs.get('rand_rotation', 0)*(np.pi/180)
            rand_shear = (tf.random.uniform([batch_size, 1]) - 0.5)*kwargs.get('rand_shear', 0)*(np.pi/180)
            rand_x_trans = (tf.random.uniform([batch_size, 1]) - 0.5)*kwargs.get('rand_x_trans', 0)*W \
                           + 0.5*W*(1.0 - rand_x_scale*tf.math.cos(rand_rotation)) \
                           + 0.5*H*rand_y_scale*tf.math.sin(rand_rotation + rand_shear)
            rand_y_trans = (tf.random.uniform([batch_size, 1]) - 0.5)*kwargs.get('rand_y_trans', 0)*H \
                           - 0.5*W*rand_x_scale*tf.math.sin(rand_rotation) \
                           + 0.5*H*(1.0 - rand_y_scale*tf.math.cos(rand_rotation + rand_shear))

            a0a = rand_x_scale*tf.math.cos(rand_rotation + rand_shear)
            a1a = -rand_y_scale*tf.math.sin(rand_rotation)
            a2a = rand_x_trans
            b0a = rand_x_scale*tf.math.sin(rand_rotation + rand_shear)
            b1a = rand_y_scale*tf.math.cos(rand_rotation)
            b2a = rand_y_trans

            rand_x_reflect = tf.math.round(tf.random.uniform([batch_size, 1])*kwargs.get('rand_x_reflect', False))
            rand_y_reflect = tf.math.round(tf.random.uniform([batch_size, 1])*kwargs.get('rand_y_reflect', False))

            a0r = 1.0 - 2.0*rand_x_reflect
            # a1r = tf.zeros([batch_size, 1], dtype=tf.float32)
            a2r = rand_x_reflect*W
            # b0r = tf.zeros([batch_size, 1], dtype=tf.float32)
            b1r = 1.0 - 2.0*rand_y_reflect
            b2r = rand_y_reflect*H

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
            self.crop_scale = kwargs.get('rand_crop_scale', (1.0, 1.0))  # Size of crop windows
            self.crop_ratio = kwargs.get('rand_crop_ratio', (1.0, 1.0))  # Aspect ratio of crop windows
            if mask is None:
                x = tf.map_fn(self.rand_crop_image, x, parallel_iterations=32, back_prop=False)
            else:
                x, mask = tf.map_fn(self.rand_crop_image_and_mask, (x, mask), dtype=(tf.float32, tf.float32),
                                    parallel_iterations=32, back_prop=False)

        return x, mask

    def rand_crop_image(self, x):
        image = x

        shape_tensor = tf.shape(image)
        H = tf.cast(shape_tensor[0], dtype=tf.int32)
        W = tf.cast(shape_tensor[1], dtype=tf.int32)

        lower, upper = self.crop_scale
        a = upper**2 - lower**2
        b = lower**2
        randval = tf.random.uniform([], dtype=tf.float32)
        rand_scale = tf.math.sqrt(a*randval + b)

        lower, upper = self.crop_ratio
        base = float(upper/lower)
        randval = tf.random.uniform([], dtype=tf.float32)
        rand_ratio = lower*tf.math.pow(base, randval)

        rand_x_scale = tf.math.sqrt(rand_scale/rand_ratio)
        rand_y_scale = tf.math.sqrt(rand_scale*rand_ratio)

        size_h = tf.cast(tf.math.round(self.input_size[0]*rand_y_scale), dtype=tf.int32)
        size_h = tf.math.minimum(H, size_h)
        size_w = tf.cast(tf.math.round(self.input_size[1]*rand_x_scale), dtype=tf.int32)
        size_w = tf.math.minimum(W, size_w)

        offset_h = tf.random.uniform([], 0, H - size_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, W - size_w + 1, dtype=tf.int32)

        image = tf.slice(image, [offset_h, offset_w, 0], [size_h, size_w, -1])
        image = tf.image.resize_bilinear(tf.expand_dims(image, axis=0), self.input_size[0:2], half_pixel_centers=True)
        image = tf.reshape(image, self.input_size)

        return image

    def rand_crop_image_and_mask(self, x):
        image = x[0]
        mask = x[1]

        shape_tensor = tf.shape(image)
        H = tf.cast(shape_tensor[0], dtype=tf.int32)
        W = tf.cast(shape_tensor[1], dtype=tf.int32)

        lower, upper = self.crop_scale
        a = upper**2 - lower**2
        b = lower**2
        randval = tf.random.uniform([], dtype=tf.float32)
        rand_scale = tf.math.sqrt(a*randval + b)

        lower, upper = self.crop_ratio
        base = float(upper / lower)
        randval = tf.random.uniform([], dtype=tf.float32)
        rand_ratio = lower*tf.math.pow(base, randval)

        rand_x_scale = tf.math.sqrt(rand_scale/rand_ratio)
        rand_y_scale = tf.math.sqrt(rand_scale*rand_ratio)

        size_h = tf.cast(tf.math.round(self.input_size[0]*rand_y_scale), dtype=tf.int32)
        size_h = tf.math.minimum(H, size_h)
        size_w = tf.cast(tf.math.round(self.input_size[1]*rand_x_scale), dtype=tf.int32)
        size_w = tf.math.minimum(W, size_w)

        offset_h = tf.random.uniform([], 0, H - size_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, W - size_w + 1, dtype=tf.int32)

        image = tf.slice(image, [offset_h, offset_w, 0], [size_h, size_w, -1])
        image = tf.image.resize_bilinear(tf.expand_dims(image, axis=0), self.input_size[0:2], half_pixel_centers=True)
        image = tf.reshape(image, self.input_size)
        mask = tf.slice(mask, [offset_h, offset_w, 0], [size_h, size_w, -1])
        mask = tf.image.resize_nearest_neighbor(tf.expand_dims(mask, axis=0),
                                                self.input_size[0:2], half_pixel_centers=True)
        mask = tf.reshape(mask, list(self.input_size[:-1]) + [1])

        return image, mask

    def center_crop(self, x):
        with tf.variable_scope('center_crop'):
            shape_tensor = tf.shape(x)
            H = tf.cast(shape_tensor[1], dtype=tf.float32)
            W = tf.cast(shape_tensor[2], dtype=tf.float32)

            offset_height = tf.cast((H - self.input_size[0])//2, dtype=tf.int32)
            offset_width = tf.cast((W - self.input_size[1])//2, dtype=tf.int32)
            target_height = tf.constant(self.input_size[0], dtype=tf.int32)
            target_width = tf.constant(self.input_size[1], dtype=tf.int32)

            x = tf.slice(x, [0, offset_height, offset_width, 0], [-1, target_height, target_width, -1])

        return x

    def rand_hue(self, x, **kwargs):
        with tf.variable_scope('rand_hue'):
            max_delta = kwargs.get('rand_hue', 0.0)

            delta = tf.random.uniform([], minval=-max_delta/2, maxval=max_delta/2, dtype=tf.float32)

            x = x + self.image_mean
            x = tf.image.adjust_hue(x, delta)
            x = x - self.image_mean

        return x

    def rand_saturation(self, x, **kwargs):
        with tf.variable_scope('rand_saturation'):
            lower, upper = kwargs.get('rand_saturation', (1.0, 1.0))

            base = float(upper / lower)
            randval = tf.random.uniform([], dtype=tf.float32)
            randval = lower*tf.math.pow(base, randval)

            x = x + self.image_mean
            x = tf.image.adjust_saturation(x, randval)
            x = x - self.image_mean

        return x

    def rand_color_balance(self, x, **kwargs):
        with tf.variable_scope('random_color_balance'):
            shape_tensor = tf.shape(x)
            lower, upper = kwargs.get('rand_color_balance', (1.0, 1.0))

            base = float(upper/lower)
            randvals = tf.random.uniform([shape_tensor[0], 1, 1, 3], dtype=tf.float32)
            randvals = lower*tf.math.pow(base, randvals)

            image_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

            x = x - image_mean
            x = x*randvals
            x = x + image_mean

        return x

    def rand_equalization(self, x, **kwargs):
        with tf.variable_scope('random_equalization'):
            shape_tensor = tf.shape(x)
            prob = kwargs.get('rand_equalization', 0.0)

            normal = tf.cast(tf.greater(tf.random.uniform([shape_tensor[0], 1, 1, 1]), prob), dtype=tf.float32)
            maxvals = tf.reduce_max(tf.math.abs(x), axis=[1, 2], keepdims=True)

            image_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

            x = x - image_mean
            x = normal*x + (1.0 - normal)*x/maxvals*0.5
            x = x + image_mean

        return x

    def rand_contrast(self, x, **kwargs):
        with tf.variable_scope('random_contrast'):
            shape_tensor = tf.shape(x)
            lower, upper = kwargs.get('rand_contrast', (1.0, 1.0))

            base = float(upper/lower)
            randvals = tf.random.uniform([shape_tensor[0], 1, 1, 1], dtype=tf.float32)
            randvals = lower*tf.math.pow(base, randvals)

            image_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

            x = x - image_mean
            x = x*randvals
            x = x + image_mean

        return x

    def rand_brightness(self, x, **kwargs):
        with tf.variable_scope('random_brightness'):
            shape_tensor = tf.shape(x)
            max_delta = kwargs.get('rand_brightness', 0.0)

            randval = tf.random.uniform([shape_tensor[0], 1, 1, 1], minval=-max_delta/2, maxval=max_delta/2,
                                        dtype=tf.float32)
            x = x + randval

        return x

    def rand_noise(self, x, **kwargs):
        with tf.variable_scope('rand_noise'):
            shape_tensor = tf.shape(x)
            noise_mean = kwargs.get('rand_noise_mean', 0.0)
            noise_stddev = kwargs.get('rand_noise_stddev', 0.0)

            noise = tf.random.normal(shape_tensor, mean=0.0, stddev=noise_stddev, dtype=tf.float32)
            noise_mask = tf.random.uniform([shape_tensor[0], 1, 1, 1], dtype=tf.float32)
            x = x + noise_mask*noise + noise_mean

        return x

    def rand_solarization(self, x, **kwargs):
        with tf.variable_scope('rand_solarization'):
            shape_tensor = tf.shape(x)
            lower, upper = kwargs.get('rand_solarization', (0.0, 1.0))

            thres_lower = tf.random.uniform([shape_tensor[0], 1, 1, 1], -0.5, lower - 0.5, dtype=tf.float32)
            thres_lower = tf.broadcast_to(thres_lower, shape_tensor)
            lower_pixels = tf.less(x, thres_lower)

            thres_upper = tf.random.uniform([shape_tensor[0], 1, 1, 1], upper - 0.5, 0.5, dtype=tf.float32)
            thres_upper = tf.broadcast_to(thres_upper, shape_tensor)
            upper_pixels = tf.greater(x, thres_upper)

            invert = tf.cast(tf.logical_or(lower_pixels, upper_pixels), dtype=tf.float32)

            x = invert*(-x) + (1.0 - invert)*x

        return x

    def rand_posterization(self, x, **kwargs):
        with tf.variable_scope('rand_posterization'):
            shape_tensor = tf.shape(x)
            lower, upper = kwargs.get('rand_posterization', (8, 8))

            factors = tf.math.round(tf.random.uniform([shape_tensor[0], 1, 1, 1],
                                                      lower - 0.5, upper + 0.5, dtype=tf.float32))
            maxvals = tf.pow(2.0, factors)
            x = tf.math.round(x*maxvals)
            x = x/maxvals

        return x

    def cutmix(self, x, y):
        with tf.variable_scope('cutmix'):
            shape_tensor = tf.shape(x)
            batch_size = shape_tensor[0]
            H = tf.cast(shape_tensor[1], tf.float32)
            W = tf.cast(shape_tensor[2], tf.float32)

            randval = tf.random.uniform([], dtype=tf.float32)
            r_h = tf.random.uniform([], 0, H, dtype=tf.float32)
            r_w = tf.random.uniform([], 0, W, dtype=tf.float32)
            size_h = H*tf.math.sqrt(1.0 - randval)
            size_w = W*tf.math.sqrt(1.0 - randval)

            hs = tf.cast(tf.math.round(tf.math.maximum(r_h - size_h/2, 0)), dtype=tf.int32)
            he = tf.cast(tf.math.round(tf.math.minimum(r_h + size_h/2, H)), dtype=tf.int32)
            ws = tf.cast(tf.math.round(tf.math.maximum(r_w - size_w/2, 0)), dtype=tf.int32)
            we = tf.cast(tf.math.round(tf.math.minimum(r_w + size_w/2, W)), dtype=tf.int32)

            m = tf.ones([1, he - hs, we - ws, 1], dtype=tf.float32)
            paddings = [[0, 0],
                        [hs, shape_tensor[1] - he],
                        [ws, shape_tensor[2] - we],
                        [0, 0]]
            m = tf.pad(m, paddings, constant_values=0.0)

            lamb = 1.0 - (tf.cast((he - hs)*(we - ws), dtype=tf.float32))/(H*W)

            idx = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)
            shuffled_x = tf.gather(x, idx, axis=0)
            shuffled_y = tf.gather(y, idx, axis=0)

            x = (1.0 - m)*x + m*shuffled_x
            y = lamb*y + (1.0 - lamb)*shuffled_y

        return x, y

    def weight_variable(self, shape, stddev=None, name='weights'):
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        with tf.device('/cpu:0'):
            if stddev is None:
                weights = tf.get_variable(name, shape, tf.float32,
                                          initializer=tf.initializers.he_normal(),
                                          trainable=trainable)
            else:
                weights = tf.get_variable(name, shape, tf.float32,
                                          tf.random_normal_initializer(mean=0.0, stddev=stddev),
                                          trainable=trainable)

            if not tf.get_variable_scope().reuse:
                tf.add_to_collection('weight_variables', weights)
                tf.add_to_collection('block{}_variables'.format(self._curr_block), weights)
                tf.add_to_collection('block{}_weight_variables'.format(self._curr_block), weights)
                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([weights]))
            weights_ema = self.ema.average(weights)
            if not tf.get_variable_scope().reuse:
                tf.add_to_collection('block{}_ema_variables'.format(self._curr_block), weights_ema)

            weights = tf.cond(self.is_train,
                              lambda: weights,
                              lambda: weights_ema)

        if self.dtype is not tf.float32:
            weights = tf.cast(weights, dtype=self.dtype)

        if self.dropout_weights:
            return tf.nn.dropout(weights, rate=self.dropout_rate_weights)
        else:
            return weights

    def bias_variable(self, shape, init_value=0.0, name='biases'):
        if self.blocks_to_train is None:
            trainable = True
        elif self._curr_block in self.blocks_to_train:
            trainable = True
        else:
            trainable = False

        with tf.device('/cpu:0'):
            biases = tf.get_variable(name, shape, tf.float32,
                                     initializer=tf.constant_initializer(value=init_value),
                                     trainable=trainable)

            if not tf.get_variable_scope().reuse:
                tf.add_to_collection('bias_variables', biases)
                tf.add_to_collection('block{}_variables'.format(self._curr_block), biases)
                tf.add_to_collection('block{}_bias_variables'.format(self._curr_block), biases)
                with tf.variable_scope(self.top_scope):
                    self.update_ops.append(self.ema.apply([biases]))
            biases_ema = self.ema.average(biases)
            if not tf.get_variable_scope().reuse:
                tf.add_to_collection('block{}_ema_variables'.format(self._curr_block), biases_ema)

            biases = tf.cond(self.is_train,
                             lambda: biases,
                             lambda: biases_ema)

        if self.dtype is not tf.float32:
            biases = tf.cast(biases, dtype=self.dtype)

        return biases

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

        if not tf.get_variable_scope().reuse:
            self._flops += out_size[0]*out_size[1]*in_channels

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

        if not tf.get_variable_scope().reuse:
            self._flops += out_size[0]*out_size[1]*in_channels

        return tf.nn.avg_pool(x, ksize=ksize, strides=strides, data_format=data_format, padding=padding)

    def conv_layer(self, x, kernel, stride, out_channels, padding='SAME', biased=True,
                   dilation=(1, 1), depthwise=False, **kwargs):
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

        weights_stddev = kwargs.get('weights_stddev', None)
        biases_value = kwargs.get('biases_value', 0.0)

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

        if depthwise:
            channel_multiplier = out_channels//in_channels
            weights = self.weight_variable([kernel[0], kernel[1], in_channels, channel_multiplier],
                                           stddev=weights_stddev,
                                           name='weights')
            convs = tf.nn.depthwise_conv2d(x, weights, strides=conv_strides, padding=padding,
                                           data_format=data_format, rate=dilation)

            if not tf.get_variable_scope().reuse:
                self._flops += out_size[0]*out_size[1]*kernel[0]*kernel[1]*in_channels*channel_multiplier
                self._params += kernel[0]*kernel[1]*in_channels*channel_multiplier
        else:
            weights = self.weight_variable([kernel[0], kernel[1], in_channels, out_channels], stddev=weights_stddev)
            convs = tf.nn.conv2d(x, weights, strides=conv_strides, padding=padding,
                                 data_format=data_format, dilations=conv_dilations)

            if not tf.get_variable_scope().reuse:
                self._flops += out_size[0]*out_size[1]*kernel[0]*kernel[1]*in_channels*out_channels
                self._params += kernel[0]*kernel[1]*in_channels*out_channels

        if biased:
            biases = self.bias_variable(out_channels, init_value=biases_value)

            if not tf.get_variable_scope().reuse:
                self._flops += out_size[0]*out_size[1]*out_channels
                self._params += out_channels

            return tf.nn.bias_add(convs, biases, data_format=data_format)
        else:
            return convs

    def fc_layer(self, x, out_dim, biased=True, **kwargs):
        weights_stddev = kwargs.get('weights_stddev', None)
        biases_value = kwargs.get('biases_value', 0.0)
        in_dim = int(x.get_shape()[-1])

        weights = self.weight_variable([in_dim, out_dim], stddev=weights_stddev)

        if not tf.get_variable_scope().reuse:
            self._flops += in_dim*out_dim
            self._params += in_dim*out_dim

        if biased:
            biases = self.bias_variable(out_dim, init_value=biases_value)

            if not tf.get_variable_scope().reuse:
                self._flops += out_dim
                self._params += out_dim

            return tf.matmul(x, weights) + biases
        else:
            return tf.matmul(x, weights)

    def batch_norm(self, x, scale=True, shift=True, is_training=None, zero_scale_init=False, scope='bn'):
        if self.train_batch_norm is not None:
            trainable = self.train_batch_norm
        else:
            if self.blocks_to_train is None:
                trainable = True
            elif self._curr_block in self.blocks_to_train:
                trainable = True
            else:
                trainable = False

        if self.channel_first:
            _, in_channels, h, w = x.get_shape().as_list()
        else:
            _, h, w, in_channels = x.get_shape().as_list()

        momentum = self.batch_norm_decay
        epsilon = 1e-4
        if is_training is None:
            is_training = self.is_train
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]

        with tf.variable_scope(scope):
            with tf.device('/cpu:0'):
                mu = tf.get_variable('mu', in_channels, dtype=tf.float32,
                                     initializer=tf.zeros_initializer(), trainable=False)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('block{}_variables'.format(self._curr_block), mu)
                    tf.add_to_collection('block{}_batch_norm_variables'.format(self._curr_block), mu)
                    with tf.variable_scope(self.top_scope):
                        self.update_ops.append(self.ema.apply([mu]))
                    self._flops += h*w*in_channels
                mu_ema = self.ema.average(mu)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('block{}_ema_variables'.format(self._curr_block), mu_ema)

                sigma = tf.get_variable('sigma', in_channels, dtype=tf.float32,
                                        initializer=tf.ones_initializer(), trainable=False)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('block{}_variables'.format(self._curr_block), sigma)
                    tf.add_to_collection('block{}_batch_norm_variables'.format(self._curr_block), sigma)
                    with tf.variable_scope(self.top_scope):
                        self.update_ops.append(self.ema.apply([sigma]))
                    self._flops += h*w*in_channels
                sigma_ema = self.ema.average(sigma)
                if not tf.get_variable_scope().reuse:
                    tf.add_to_collection('block{}_ema_variables'.format(self._curr_block), sigma_ema)

                if shift:
                    beta = tf.get_variable('beta', in_channels, dtype=tf.float32,
                                           initializer=tf.zeros_initializer(), trainable=trainable)
                    if not tf.get_variable_scope().reuse:
                        tf.add_to_collection('block{}_variables'.format(self._curr_block), beta)
                        tf.add_to_collection('block{}_batch_norm_variables'.format(self._curr_block), beta)
                        with tf.variable_scope(self.top_scope):
                            self.update_ops.append(self.ema.apply([beta]))
                        self._flops += h*w*in_channels
                        self._params += in_channels
                    beta_ema = self.ema.average(beta)
                    if not tf.get_variable_scope().reuse:
                        tf.add_to_collection('block{}_ema_variables'.format(self._curr_block), beta_ema)
                else:
                    beta = None
                    beta_ema = None

                if scale:
                    scale_initializer = tf.zeros_initializer() if zero_scale_init else tf.ones_initializer()
                    gamma = tf.get_variable('gamma', in_channels, dtype=tf.float32,
                                            initializer=scale_initializer, trainable=trainable)
                    if not tf.get_variable_scope().reuse:
                        tf.add_to_collection('block{}_variables'.format(self._curr_block), gamma)
                        tf.add_to_collection('block{}_batch_norm_variables'.format(self._curr_block), gamma)
                        with tf.variable_scope(self.top_scope):
                            self.update_ops.append(self.ema.apply([gamma]))
                        self._flops += h*w*in_channels
                        self._params += in_channels
                    gamma_ema = self.ema.average(gamma)
                    if not tf.get_variable_scope().reuse:
                        tf.add_to_collection('block{}_ema_variables'.format(self._curr_block), gamma_ema)
                else:
                    gamma = None
                    gamma_ema = None

                mean, var, beta, gamma = tf.cond(is_training,
                                                 lambda: (mu, sigma, beta, gamma),
                                                 lambda: (mu_ema, sigma_ema, beta_ema, gamma_ema))

            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=tf.float32)
            data_format = 'NCHW' if self.channel_first else 'NHWC'
            x, batch_mean, batch_var = tf.cond(is_training,
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
            if self.dtype is not tf.float32:
                x = tf.cast(x, dtype=self.dtype)

            update = 1.0 - momentum
            update_mu = mu.assign(momentum*mu + update*batch_mean)
            update_sigma = sigma.assign(momentum*sigma + update*batch_var)
            if trainable:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        return x

    def upsampling_2d_layer(self, x, scale=2, name='upsampling'):
        if self.channel_first:
            x = tf.transpose(x, perm=[0, 2, 3, 1], name='tp')
        in_shape = x.get_shape()
        x = tf.image.resize_bilinear(x, [in_shape[1]*scale, in_shape[2]*scale], align_corners=False, name=name)
        if self.channel_first:
            x = tf.transpose(x, perm=[0, 3, 1, 2], name='tp')

        return x

    def transposed_conv_layer(self, x, kernel, stride, out_channels, padding='SAME', biased=True,
                              dilation=(1, 1), **kwargs):
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
            if padding.lower() == 'valid':
                output_shape = [batch_size, h*stride[0] - kernel[0] + 1, w*stride[1] - kernel[1] + 1, out_channels]
            else:
                output_shape = [batch_size, h*stride[0], w*stride[1], out_channels]
            out_size = output_shape[1:3]

        weights_stddev = kwargs.get('weights_stddev', None)
        biases_value = kwargs.get('biases_value', 0.0)

        weights = self.weight_variable([kernel[0], kernel[1], in_channels, out_channels], stddev=weights_stddev)
        convs = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape, strides=conv_strides,
                                       padding=padding, data_format=data_format, dilations=conv_dilations)

        if not tf.get_variable_scope().reuse:
            self._flops += out_size[0]*out_size[1]*kernel[0]*kernel[1]*in_channels*out_channels
            self._params += kernel[0]*kernel[1]*in_channels*out_channels

        if biased:
            biases = self.bias_variable(out_channels, init_value=biases_value)

            if not tf.get_variable_scope().reuse:
                self._flops += out_size[0]*out_size[1]*out_channels
                self._params += out_channels

            return tf.nn.bias_add(convs, biases, data_format=data_format)
        else:
            return convs

    def relu(self, x, name='relu'):
        if not tf.get_variable_scope().reuse:
            shape = x.get_shape().as_list()
            self._flops += np.prod(shape[1:])

        return tf.nn.relu(x, name=name)

    def swish(self, x, name='swish'):
        if not tf.get_variable_scope().reuse:
            shape = x.get_shape().as_list()
            self._flops += 2*np.prod(shape[1:])

        return tf.nn.swish(x, name=name)

    def sigmoid(self, x, name=None):
        if not tf.get_variable_scope().reuse:
            shape = x.get_shape().as_list()
            self._flops += np.prod(shape[1:])

        return tf.nn.sigmoid(x, name=name)

    def grad_cam(self, logits, conv_layer, y=None):
        eps = 1e-4
        with tf.name_scope('grad_cam'):
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
            gcam = tf.image.resize_bilinear(gcam, self.input_size[0:2], align_corners=False)
            gcam = gcam/(tf.reduce_max(gcam, axis=[1, 2, 3], keepdims=True) + eps)

            return gcam
