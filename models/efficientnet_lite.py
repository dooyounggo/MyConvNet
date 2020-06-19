"""
EfficientNet-Lite
https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html
"""
import tensorflow.compat.v1 as tf
import numpy as np
from convnet import ConvNet


class EfficientNetLite(ConvNet):
    def _init_params(self, **kwargs):
        self.channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        self.kernels = [3, 3, 3, 5, 3, 5, 5, 3, None]
        self.strides = [2, 1, 2, 2, 2, 1, 2, 1, None]
        self.conv_units = [None, 1, 2, 2, 3, 3, 4, 1, None]
        self.multipliers = [None, 1, 6, 6, 6, 6, 6, 6, None]

        self.conv_initializer = tf.initializers.variance_scaling(mode='fan_out')
        self.fc_initializer = tf.initializers.variance_scaling(scale=1.0/3.0,
                                                               mode='fan_out',
                                                               distribution='uniform')

        self.striding_kernel_offset = kwargs.get('striding_kernel_offset', 0)
        self.striding_kernel_size = kwargs.get('striding_kernel_size', None)

        self.initial_drop_rate = kwargs.get('initial_drop_rate', 0.0)
        self.final_drop_rate = kwargs.get('final_drop_rate', 0.0)

    def _build_model(self):
        d = dict()

        X_input = self.X

        channels = self.channels
        kernels = self.kernels
        strides = self.strides
        conv_units = self.conv_units
        multipliers = self.multipliers

        channels[0] = 32  # Fix stem
        channels[-1] = 1280  # Fix head
        conv_units[-2] = 1  # Fix last mbconv

        len_c = len(channels)
        len_k = len(kernels)
        len_s = len(strides)
        len_r = len(conv_units)
        len_m = len(multipliers)
        num_blocks = min([len_c, len_k, len_s, len_r, len_m])

        self._curr_block = 0
        with tf.variable_scope('block_0'):
            with tf.variable_scope('conv_0'):
                k = kernels[0]
                if strides[0] > 1:
                    if self.striding_kernel_size is None:
                        k += self.striding_kernel_offset
                    else:
                        k = self.striding_kernel_size
                x = self.conv_layer(X_input, k, strides[0], channels[0], padding='SAME', biased=False,
                                    weight_initializer=self.conv_initializer, verbose=True)
                d['block_0' + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='norm')
                d['block_0' + '/conv_0' + '/norm'] = x
                x = self.relu6(x, name='relu6')
                d['block_0' + '/conv_0' + '/relu6'] = x
            d['block_0'] = x

        for i in range(1, num_blocks - 1):
            self._curr_block = i
            dr = self.initial_drop_rate + (self.final_drop_rate - self.initial_drop_rate)*i/(num_blocks - 2)
            print('block {} drop rate = {:.3f}'.format(i, dr))
            for j in range(conv_units[i]):
                k = kernels[i]
                if j > 0:
                    s = 1
                else:
                    s = strides[i]
                    if s > 1:
                        if self.striding_kernel_size is None:
                            k += self.striding_kernel_offset
                        else:
                            k = self.striding_kernel_size
                x = self._mb_conv_unit(x, k, s, channels[i], multipliers[i], d, drop_rate=dr,
                                       name='block_{}/mbconv_{}'.format(i, j))
            d['block_{}'.format(self._curr_block)] = x

        self._curr_block += 1
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 1, 1, self.channels[-1], padding='SAME', biased=False, depthwise=False,
                                    weight_initializer=self.conv_initializer, verbose=True)
                d['logits' + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='norm')
                d['logits' + '/conv_0' + '/norm'] = x
                x = self.relu6(x, name='relu6')
                d['logits' + '/conv_0' + '/relu6'] = x
        d['block_{}'.format(self._curr_block)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x

                    if self.feature_reduction > 1:
                        with tf.variable_scope('comp'):  # Feature compression (experimental)
                            num_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
                            x = self.fc_layer(x, num_channels//self.feature_reduction)
                            x = self.relu6(x, name='relu6')

                    x = tf.nn.dropout(x, rate=self.dropout_rate_features)
                    x = self.fc_layer(x, self.num_classes,
                                      weight_initializer=self.fc_initializer)

                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _mb_conv_unit(self, x, kernel, stride, out_channels, multiplier, d, drop_rate=0.0, name='mbconv'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            if stride[0] == 1 and stride[1] == 1 and in_channels == out_channels:
                skip = x
            else:
                skip = None
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                if multiplier > 1:
                    x = self.conv_layer(x, 1, 1, in_channels*multiplier, padding='SAME', biased=False, depthwise=False,
                                        weight_initializer=self.conv_initializer, verbose=True)
                    d[name + '/conv_0'] = x
                    x = self.batch_norm(x, shift=True, scale=True, scope='norm')
                    d[name + '/conv_0' + '/norm'] = x
                    x = self.relu6(x, name='relu6')
                    d[name + '/conv_0' + '/relu6'] = x

            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernel, stride, in_channels*multiplier,
                                    padding='SAME', biased=False, depthwise=True,
                                    weight_initializer=self.conv_initializer, verbose=True)
                d[name + '/conv_1'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='norm')
                d[name + '/conv_1' + '/norm'] = x
                x = self.relu6(x, name='relu6')
                d[name + '/conv_1' + '/relu6'] = x

            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=False, depthwise=False,
                                    weight_initializer=self.conv_initializer, verbose=True)
                d[name + '/conv_2'] = x
                x = self.batch_norm(x, shift=True, scale=True, zero_scale_init=skip is not None, scope='norm')
                d[name + '/conv_2' + '/norm'] = x

            if skip is not None:
                x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            d[name] = x

        return x

    def _calc_widths(self, widths, coefficient):
        divisor = 8
        new_widths = []
        for w in widths:
            if w is None:
                new_widths.append(None)
            else:
                w = coefficient*w
                new_w = max(divisor, (int(w + divisor/2)//divisor)*divisor)
                if new_w < 0.9*w:
                    new_w += divisor
                new_widths.append(new_w)
        return new_widths

    def _calc_depths(self, depths, coefficient):
        new_depths = []
        for d in depths:
            if d is None:
                new_depths.append(None)
            else:
                new_depths.append(int(np.ceil(coefficient*d)))
        return new_depths


class EfficientNetLite0(EfficientNetLite):  # 224
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        print('Widths:', self.channels)
        print('Depths:', self.conv_units)


class EfficientNetLite1(EfficientNetLite):  # 240
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.channels = self._calc_widths(self.channels, 1.0)
        self.conv_units = self._calc_depths(self.conv_units, 1.1)
        print('Widths:', self.channels)
        print('Depths:', self.conv_units)


class EfficientNetLite2(EfficientNetLite):  # 260
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.channels = self._calc_widths(self.channels, 1.1)
        self.conv_units = self._calc_depths(self.conv_units, 1.2)
        print('Widths:', self.channels)
        print('Depths:', self.conv_units)


class EfficientNetLite3(EfficientNetLite):  # 300
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.channels = self._calc_widths(self.channels, 1.2)
        self.conv_units = self._calc_depths(self.conv_units, 1.4)
        print('Widths:', self.channels)
        print('Depths:', self.conv_units)


class EfficientNetLite4(EfficientNetLite):  # 380
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.channels = self._calc_widths(self.channels, 1.4)
        self.conv_units = self._calc_depths(self.conv_units, 1.8)
        print('Widths:', self.channels)
        print('Depths:', self.conv_units)
