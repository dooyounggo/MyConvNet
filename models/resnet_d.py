import tensorflow.compat.v1 as tf
from convnet import ConvNet


class ResNetD(ConvNet):  # Base model. ResNet-D-50
    def _init_params(self, **kwargs):
        self.channels = [64, 256, 512, 1024, 2048]
        self.kernels = [3, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 2]
        self.res_units = [None, 3, 4, 6, 3]

        self.erase_relu = kwargs.get('erase_relu', False)
        self.activation_type = kwargs.get('activation_type', 'relu')
        self.striding_kernel_offset = kwargs.get('striding_kernel_offset', 0)
        self.erase_max_pool = kwargs.get('erase_max_pool', False)
        if self.erase_max_pool:
            self.strides[1] = 2

        self.initial_drop_rate = kwargs.get('initial_drop_rate', 0.0)
        self.final_drop_rate = kwargs.get('final_drop_rate', 0.0)

    def _build_model(self):
        d = dict()

        X_input = self.X

        channels = self.channels
        kernels = self.kernels
        strides = self.strides
        res_units = self.res_units

        len_c = len(channels)
        len_k = len(kernels)
        len_s = len(strides)
        len_r = len(res_units) + 1
        self._num_blocks = min([len_c, len_k, len_s, len_r])

        with tf.variable_scope('block_0'):
            with tf.variable_scope('conv_0'):
                k = kernels[0] + self.striding_kernel_offset
                x = self.conv_layer(X_input, k, strides[0], channels[0]//2, padding='SAME', biased=False,
                                    verbose=True)
                d['block_0' + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_0' + '/bn'] = x
                x = self.activation(x, activation_type=self.activation_type)
                d['block_0' + '/conv_0' + '/' + self.activation_type] = x
            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernels[0], 1, channels[0]//2, padding='SAME', biased=False, verbose=True)
                d['block_0' + '/conv_1'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_1' + '/bn'] = x
                x = self.activation(x, activation_type=self.activation_type)
                d['block_0' + '/conv_1' + '/' + self.activation_type] = x
            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, kernels[0], 1, channels[0], padding='SAME', biased=False, verbose=True)
                d['block_0' + '/conv_2'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_2' + '/bn'] = x
                if not self.erase_relu:
                    x = self.activation(x, activation_type=self.activation_type)
            if not self.erase_max_pool:
                x = self.max_pool(x, 2, 2, padding='SAME')
            d['block_0'] = x

        for i in range(1, self.num_blocks):
            self._curr_block = i
            dr = self.initial_drop_rate + (self.final_drop_rate - self.initial_drop_rate)*i/(self.num_blocks - 1)
            print('block {} drop rate = {:.3f}'.format(i, dr))
            for j in range(res_units[i]):
                k = kernels[i]
                if j > 0:
                    s = 1
                else:
                    s = strides[i]
                    if s == 2:
                        k += self.striding_kernel_offset
                x = self._res_unit(x, k, s, channels[i], d, drop_rate=dr, name='block_{}/res_{}'.format(i, j))
            d['block_{}'.format(i)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    if self.erase_relu:
                        x = self.activation(x, activation_type=self.activation_type)
                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x
                    x = tf.nn.dropout(x, rate=self.dropout_rate_features)
                    x = self.fc_layer(x, self.num_classes)
                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            if in_channels == out_channels:
                if stride[0] > 1 | stride[1] > 1:
                    skip = self.avg_pool(x, stride, stride, padding='SAME')
                else:
                    skip = x
            else:
                with tf.variable_scope('conv_skip'):
                    skip = self.avg_pool(x, stride, stride, padding='SAME')
                    skip = self.conv_layer(skip, 1, 1, out_channels, padding='SAME', biased=False, verbose=True)
                    skip = self.batch_norm(skip, shift=True, scale=True, scope='bn')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 1, 1, out_channels//4, padding='SAME', biased=False, verbose=True)
                d[name + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                x = self.activation(x, activation_type=self.activation_type)
                d[name + '/conv_0' + '/' + self.activation_type] = x

            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernel, stride, out_channels//4, padding='SAME', biased=False, verbose=True)
                d[name + '/conv_1'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.activation(x, activation_type=self.activation_type)
                d[name + '/conv_1' + '/' + self.activation_type] = x

            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=False, verbose=True)
                d[name + '/conv_2'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn', zero_scale_init=True)
                d[name + '/conv_2' + '/bn'] = x

            if skip is not None:
                x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            if not self.erase_relu:
                x = self.activation(x, activation_type=self.activation_type)
            d[name] = x

        return x


class ResNetD50(ResNetD):
    pass


class ResNetD101(ResNetD):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.res_units = [None, 3, 4, 23, 3]
