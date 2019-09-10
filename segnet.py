"""
==================== Reference Pages ====================
GCN: https://arxiv.org/abs/1703.02719
"""

import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from convnet import ConvNet


class SegNet(ConvNet):
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
                self.dropout_rate_weights = tf.cond(tf.constant(self.dropout_weights, dtype=tf.bool),
                                                    lambda: self.dropout_rate,
                                                    lambda: tf.constant(0.0, dtype=self.dtype),
                                                    name='dropout_rate_weights')
                self.dropout_rate_logits = tf.cond(tf.constant(self.dropout_logits, dtype=tf.bool),
                                                   lambda: self.dropout_rate,
                                                   lambda: tf.constant(0.0, dtype=self.dtype),
                                                   name='dropout_rate_logits')
                self.image_mean = tf.cond(tf.constant(kwargs.get('zero_center'), dtype=tf.bool, name='zero_center'),
                                          lambda: tf.constant(kwargs.get('image_mean', 0.5), dtype=tf.float32),
                                          lambda: tf.constant(0.0, dtype=tf.float32),
                                          name='image_mean')

        self.X_in = []
        self.Y_in = []
        if self.image_size is None:
            output_shapes = ([None, None, None, self.input_size[-1]],
                             None)
        else:
            output_shapes = ([None, self.image_size[0], self.image_size[1], self.input_size[-1]],
                             None)
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
                        self._broadcast_shape = [tf.shape(self.X)[1], tf.shape(self.X)[2]]
                        self.Y = tf.map_fn(self._broadcast_nans, self.Y)  # Broadcasting for NaNs
                        self.Y = tf.where(tf.is_nan(self.Y),
                                          tf.constant(0.0, dtype=tf.float32) - tf.ones_like(self.Y, dtype=tf.float32),
                                          self.Y)

                        self.Y = tf.cast(self.Y, dtype=tf.int32)
                        self.Y = tf.one_hot(self.Y, depth=self.num_classes, dtype=tf.float32)  # one-hot encoding

                        self.X = self.zero_pad(self.X)
                        self.Y = self.zero_pad(self.Y)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        self.X, self.Y = tf.cond(self.augmentation,
                                                 lambda: self.augment_images(self.X, mask=self.Y, **kwargs),
                                                 lambda: (self.center_crop(self.X), self.center_crop(self.Y)),
                                                 name='augmentation')
                        self.Xs.append(self.X)
                        self.Ys.append(self.Y)

                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])
                            self.Y = tf.transpose(self.Y, perm=[0, 3, 1, 2])

                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.X = tf.cast(self.X, dtype=self.dtype)
                        self.backbone_only = True
                        d_backbone = self._build_model(**kwargs)
                        self.backbone_only = False
                        self.d = self._build_model_seg(d_backbone, **kwargs)
                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.d['logits'] = tf.cast(self.d['logits'], dtype=tf.float32)
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        if self.channel_first:
                            self.d['pred'] = tf.transpose(self.d['pred'], perm=[0, 2, 3, 1])
                        tf.get_variable_scope().reuse_variables()

                        self.d.update(d_backbone)
                        self.dicts.append(self.d)

                        self.logits = self.d['logits']
                        self.preds.append(self.d['pred'])
                        self.losses.append(self._build_loss(**kwargs))

                        self.bytes_in_use.append(tf.contrib.memory_stats.BytesInUse())

        with tf.device('/cpu:0'):
            with tf.variable_scope('calc'):
                self.X_all = tf.concat(self.Xs, axis=0)
                self.Y_all = tf.concat(self.Ys, axis=0)
                self.pred = tf.concat(self.preds, axis=0)
                self.loss = tf.reduce_mean(self.losses)

                self.input_images = tf.concat(self.X_in, axis=0, name='x_in')
                self.debug_images_0 = self.seg_labels_to_images(self.Y_all)
                self.debug_images_1 = self.seg_labels_to_images(self.pred)

    @abstractmethod
    def _build_model_seg(self, d_backbone, **kwargs):
        """
        Build model of segmentation networks.
        This should be implemented.
        """
        pass

    def _broadcast_nans(self, y):
        return tf.broadcast_to(y, self._broadcast_shape)

    def seg_labels_to_images(self, y):
        edge_color = 1.0
        code_r = [1, 0, 0, 1, 1, 0, .8, 1, .6, .6]
        code_g = [0, 1, 0, 1, 0, 1, .8, .6, 1, .6]
        code_b = [0, 0, 1, 0, 1, 1, .8, .6, .6, 1]
        code_length = len(code_r)
        color_base = (self.num_classes - 1)//code_length + 1
        color_coeff = edge_color/color_base

        class_inds = np.arange(self.num_classes, dtype=np.float32)
        reds = (class_inds + code_length - 1)//code_length*color_coeff
        greens = (class_inds + code_length - 1)//code_length*color_coeff
        blues = (class_inds + code_length - 1)//code_length*color_coeff
        for i in range(1, self.num_classes):
            idx = (i - 1) % code_length
            reds[i] = reds[i]*code_r[idx]
            greens[i] = greens[i]*code_g[idx]
            blues[i] = blues[i]*code_b[idx]

        with tf.variable_scope('seg_labels_to_images'):
            ignore = tf.equal(tf.reduce_sum(y, axis=-1, keepdims=True), 0.0)
            r = tf.reduce_sum(y*reds, axis=-1, keepdims=True)
            g = tf.reduce_sum(y*greens, axis=-1, keepdims=True)
            b = tf.reduce_sum(y*blues, axis=-1, keepdims=True)
            y = tf.concat([r, g, b], axis=-1) + tf.cast(ignore, dtype=tf.float32)*edge_color

        return y
