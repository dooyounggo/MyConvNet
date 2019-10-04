import tensorflow as tf
import numpy as np
from convnet import ConvNet


class EfficientNet(ConvNet):
    def _init_params(self):
        self.channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        self.kernels = [3, 3, 3, 5, 3, 5, 5, 3]
        self.strides = [2, 1, 2, 2, 2, 1, 2, 1]
        self.res_units = [1, 2, 2, 3, 3, 4, 1]
        self.multipliers = [1, 6, 6, 6, 6, 6, 6]

        self.se_reduction = 4

    def _build_model(self, **kwargs):
        d = dict()

        initial_drop_rate = kwargs.get('initial_drop_rate', 0.0)
        final_drop_rate = kwargs.get('final_drop_rate', 0.0)

        X_input = self.X

        channels = self.channels
        kernels = self.kernels
        strides = self.strides
        res_units = self.res_units
        multipliers = self.multipliers

        len_c = len(channels) - 1
        len_k = len(kernels)
        len_s = len(strides)
        len_r = len(res_units) + 1
        len_m = len(multipliers) + 1
        self._num_blocks = min([len_c, len_k, len_s, len_r, len_m])

        with tf.variable_scope('block_0'):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(X_input, kernels[0], strides[0], channels[0], padding='SAME', biased=False)
                print('block_0' + '/conv_0.shape', x.get_shape().as_list())
                d['block_0' + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_0' + '/bn'] = x
                x = self.swish(x, name='swish')
                d['block_0' + '/conv_0' + '/swish'] = x
            d['block_0'] = x

        for i in range(1, self.num_blocks):
            self._curr_block = i
            dr = initial_drop_rate + (final_drop_rate - initial_drop_rate)*i/(self.num_blocks - 1)
            print('block {} drop rate = {:.3f}'.format(i, dr))
            for j in range(res_units[i-1]):
                if j > 0:
                    s = 1
                else:
                    s = strides[i]
                x = self._res_unit(x, kernels[i], s, channels[i], multipliers[i - 1], d,
                                   drop_rate=dr, name='block_{}/res_{}'.format(i, j))
            d['block_{}'.format(self._curr_block)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    with tf.variable_scope('conv_0'):
                        x = self.conv_layer(x, 1, 1, self.channels[-1], padding='SAME', biased=False, depthwise=False)
                        print('logits' + '/conv_0.shape', x.get_shape().as_list())
                        d['logits' + '/conv_0'] = x
                        x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                        d['logits' + '/conv_0' + '/bn'] = x
                        x = self.swish(x, name='swish')
                        d['logits' + '/conv_0' + '/swish'] = x

                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x

                    x = tf.nn.dropout(x, rate=self.dropout_rate_logits)
                    x = self.fc_layer(x, self.num_classes)

                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, multiplier, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            if stride[0] == 1 and stride[1] == 1:
                if in_channels != out_channels:
                    with tf.variable_scope('conv_skip'):
                        skip = self.conv_layer(x, 1, 1, out_channels, padding='SAME')
                else:
                    skip = x
            else:
                skip = None
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                if multiplier > 1:
                    x = self.conv_layer(x, 1, 1, in_channels*multiplier, padding='SAME', biased=False, depthwise=False)
                    print(name + '/conv_0.shape', x.get_shape().as_list())
                    d[name + '/conv_0'] = x
                    x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                    d[name + '/conv_0' + '/bn'] = x
                    x = self.swish(x, name='swish')
                    d[name + '/conv_0' + '/swish'] = x
                else:
                    x = tf.identity(x)

            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernel, stride, in_channels*multiplier,
                                    padding='SAME', biased=False, depthwise=True)
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.swish(x, name='swish')
                d[name + '/conv_1' + '/swish'] = x

            se_mask = self._se_mask(x, multiplier*self.se_reduction, name='se_mask')
            d[name + '/se_mask'] = se_mask
            x = x*se_mask

            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=False, depthwise=False)
                print(name + '/conv_2.shape', x.get_shape().as_list())
                d[name + '/conv_2'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_2' + '/bn'] = x

            if skip is not None:
                x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            d[name] = x

        return x

    def _se_mask(self, x, reduction, name='se_mask'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        axis = [2, 3] if self.channel_first else [1, 2]
        with tf.variable_scope(name):
            x = tf.reduce_mean(x, axis=axis)

            with tf.variable_scope('fc_0'):
                x = self.fc_layer(x, in_channels//reduction)

            x = self.swish(x, name='swish')

            with tf.variable_scope('fc_1'):
                x = self.fc_layer(x, in_channels)

            x = self.sigmoid(x)
            batch_size = tf.shape(x)[0]
            shape = [batch_size, in_channels, 1, 1] if self.channel_first else [batch_size, 1, 1, in_channels]
            x = tf.reshape(x, shape=shape)

        return x

    def _calc_widths(self, widths, coefficient):
        divisor = 8
        new_widths = []
        for w in widths:
            w = coefficient*w
            new_w = max(divisor, (int(w + divisor/2)//divisor)*divisor)
            if new_w < 0.9*w:
                new_w += divisor
            new_widths.append(new_w)
        return new_widths

    def _calc_depths(self, depths, coefficient):
        new_depths = []
        for d in depths:
            new_depths.append(int(np.ceil(coefficient*d)))
        return new_depths


class EfficientNetB0(EfficientNet):  # 224
    def _init_params(self):
        super()._init_params()
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB1(EfficientNet):  # 240
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 1.0)
        self.res_units = self._calc_depths(self.res_units, 1.1)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB2(EfficientNet):  # 260
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 1.1)
        self.res_units = self._calc_depths(self.res_units, 1.2)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB3(EfficientNet):  # 300
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 1.2)
        self.res_units = self._calc_depths(self.res_units, 1.4)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB4(EfficientNet):  # 380
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 1.4)
        self.res_units = self._calc_depths(self.res_units, 1.8)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB5(EfficientNet):  # 456
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 1.6)
        self.res_units = self._calc_depths(self.res_units, 2.2)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB6(EfficientNet):  # 528
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 1.8)
        self.res_units = self._calc_depths(self.res_units, 2.6)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)


class EfficientNetB7(EfficientNet):  # 600
    def _init_params(self):
        super()._init_params()
        self.channels = self._calc_widths(self.channels, 2.0)
        self.res_units = self._calc_depths(self.res_units, 3.1)
        print('Widths:', self.channels)
        print('Depths:', self.res_units)
