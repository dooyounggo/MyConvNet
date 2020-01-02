"""
Build segmentation networks using TensorFlow low-level APIs.
"""

import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from convnet import ConvNet


class SegNet(ConvNet):
    def _init_model(self, **kwargs):
        output_shapes = ([None, None, None, self.input_size[-1]],
                         None)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.gpu_offset, self.num_gpus + self.gpu_offset):
                self._curr_device = i
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
                        self.Y = tf.map_fn(self.broadcast_nans, self.Y)  # Broadcasting for NaNs
                        self.Y = tf.where(tf.is_nan(self.Y), tf.zeros_like(self.Y), self.Y)

                        self.Y = tf.expand_dims(self.Y, axis=-1)  # Attach the channel dimension for augmentation

                        self.X = self.zero_pad(self.X)
                        self.Y = self.zero_pad(self.Y)
                        self.X = tf.math.subtract(self.X, self.image_mean, name='zero_center')
                        self.X, self.Y = tf.cond(self.augmentation,
                                                 lambda: self.augment_images(self.X, mask=self.Y, **kwargs),
                                                 lambda: (self.center_crop(self.X), self.center_crop(self.Y)),
                                                 name='augmentation')
                        if kwargs.get('cutmix', False):
                            self._cutmix_scheduling = kwargs.get('cutmix_scheduling', False)
                            self.X, self.Y = tf.cond(self.is_train,
                                                     lambda: self.cutmix(self.X, self.Y),
                                                     lambda: (self.X, self.Y),
                                                     name='cutmix')
                        self.Y = tf.cast(tf.math.round(self.Y[..., 0] - 1.0), dtype=tf.int32)  # -1 for pixels to ignore
                        self.Y = tf.one_hot(self.Y, depth=self.num_classes, dtype=tf.float32)  # one-hot encoding
                        self.Xs.append(self.X)
                        self.Ys.append(self.Y)
                        
                        self.X *= 2  # Set input range in [-1 1]

                        if self.channel_first:
                            self.X = tf.transpose(self.X, perm=[0, 3, 1, 2])

                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.X = tf.cast(self.X, dtype=self.dtype)
                        with tf.name_scope('nn'):
                            self.backbone_only = True
                            d_backbone = self._build_model(**kwargs)
                            self.backbone_only = False
                            self.d = self._build_model_seg(d_backbone, **kwargs)
                        if self.dtype is not tf.float32:
                            with tf.name_scope('gpu{}/cast/'.format(i)):
                                self.d['logits'] = tf.cast(self.d['logits'], dtype=tf.float32)
                                self.d['pred'] = tf.cast(self.d['pred'], dtype=tf.float32)
                        tf.get_variable_scope().reuse_variables()

                        self.d.update(d_backbone)
                        self.dicts.append(self.d)
                        self.logits = self.d['logits']
                        self.pred = self.d['pred']
                        self.losses.append(self._build_loss(**kwargs))
                        self.preds.append(self.pred)

                        self.bytes_in_use.append(tf.contrib.memory_stats.BytesInUse())

        with tf.device(self.param_device):
            with tf.variable_scope('calc/'):
                self.X_all = tf.concat(self.Xs, axis=0, name='x') + self.image_mean
                self.Y_all = tf.concat(self.Ys, axis=0, name='y_true')
                self.valid_mask = tf.concat(self.valid_masks, axis=0, name='valid_mask')
                self.pred = tf.concat(self.preds, axis=0, name='y_pred')
                self.loss = tf.reduce_mean(self.losses, name='mean_loss')

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

    def label_smoothing(self, ls_factor, name='label_smoothing'):
        with tf.variable_scope(name):
            ls_factor = tf.constant(ls_factor, dtype=tf.float32, name='label_smoothing_factor')
            avg_labels = tf.nn.avg_pool2d(self.Y, (5, 5), (1, 1), padding='SAME')
            labels = (1.0 - ls_factor)*self.Y + ls_factor*avg_labels
        return labels

    def broadcast_nans(self, y):
        return tf.broadcast_to(y, self._broadcast_shape)

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

            idx = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)
            shuffled_x = tf.gather(x, idx, axis=0)
            shuffled_y = tf.gather(y, idx, axis=0)

            x = (1.0 - m)*x + m*shuffled_x
            y = (1.0 - m)*y + m*shuffled_y

        return x, y

    def seg_labels_to_images(self, y):
        max_color = 1.0
        code_r = [1, 0, 0, 1, 1, 0, .8, 1, .6, .6]
        code_g = [0, 1, 0, 1, 0, 1, .8, .6, 1, .6]
        code_b = [0, 0, 1, 0, 1, 1, .8, .6, .6, 1]
        code_length = len(code_r)
        color_base = (self.num_classes - 1)//code_length + 1
        color_coeff = max_color/color_base

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
            y = tf.concat([r, g, b], axis=-1) + tf.cast(ignore, dtype=tf.float32)*max_color

        return y
