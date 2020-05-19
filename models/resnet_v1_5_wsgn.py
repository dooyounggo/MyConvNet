import tensorflow.compat.v1 as tf
from convnet import ConvNet


class ResNet(ConvNet):  # Base model. ResNet-50 with weight standardization and group normalization
    def _init_params(self, **kwargs):
        self.channels = [64, 256, 512, 1024, 2048]
        self.kernels = [7, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 2]
        self.res_units = [None, 3, 4, 6, 3]

        self.dilations = [None, 1, 1, 1, 1]
        self.multi_grid = [1, 2, 4]

        self.erase_relu = kwargs.get('erase_relu', False)

    def _build_model(self, **kwargs):
        d = dict()

        initial_drop_rate = kwargs.get('initial_drop_rate', 0.0)
        final_drop_rate = kwargs.get('final_drop_rate', 0.0)

        X_input = self.X

        channels = self.channels
        kernels = self.kernels
        strides = self.strides
        res_units = self.res_units
        dilation = self.dilations

        len_c = len(channels)
        len_k = len(kernels)
        len_s = len(strides)
        len_r = len(res_units)
        len_d = len(dilation)
        self._num_blocks = min([len_c, len_k, len_s, len_r, len_d])

        with tf.variable_scope('block_0'):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(X_input, kernels[0], strides[0], channels[0], padding='SAME', biased=False,
                                    ws=True)
                print('block_0' + '/conv_0.shape', x.get_shape().as_list())
                d['block_0' + '/conv_0'] = x
                x = self.group_norm(x, num_groups=32, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_0' + '/gn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_0' + '/relu'] = x
                x = self.max_pool(x, 3, 2, padding='SAME')
                d['block_0' + '/conv_0' + '/maxpool'] = x
            d['block_0'] = x

        for i in range(1, self.num_blocks):
            self._curr_block = i
            dr = initial_drop_rate + (final_drop_rate - initial_drop_rate)*i/(self.num_blocks - 1)
            print('block {} drop rate = {:.3f}'.format(i, dr))
            for j in range(res_units[i]):
                if j > 0:
                    s = 1
                else:
                    s = strides[i]
                if dilation[i] == 1:
                    dil = 1
                else:
                    mg_idx = j % len(self.multi_grid)
                    dil = dilation[i]*self.multi_grid[mg_idx]
                x = self._res_unit(x, kernels[i], s, channels[i], dil,
                                   d, drop_rate=dr, name='block_{}/res_{}'.format(i, j))
            d['block_{}'.format(i)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    if self.erase_relu:
                        x = self.relu(x, name='relu')
                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x
                    x = tf.nn.dropout(x, rate=self.dropout_rate_features)
                    x = self.fc_layer(x, self.num_classes)
                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, dilation, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            if in_channels == out_channels:
                if stride[0] > 1 or stride[1] > 1:
                    skip = self.max_pool(x, stride, stride, padding='VALID')
                else:
                    skip = x
            else:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(x, 1, stride, out_channels, padding='SAME', biased=False, ws=True)
                    skip = self.group_norm(skip, num_groups=32, shift=True, scale=True, scope='bn')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 1, 1, out_channels//4, padding='SAME', biased=False, ws=True)
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x
                x = self.group_norm(x, num_groups=32, shift=True, scale=True, scope='bn')
                d[name + '/conv_0' + '/gn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_0' + '/relu'] = x

            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernel, stride, out_channels//4, padding='SAME', biased=False,
                                    dilation=dilation, ws=True)
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x
                x = self.group_norm(x, num_groups=32, shift=True, scale=True, scope='bn')
                d[name + '/conv_1' + '/gn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x

            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=False, ws=True)
                print(name + '/conv_2.shape', x.get_shape().as_list())
                d[name + '/conv_2'] = x
                x = self.group_norm(x, num_groups=32, shift=True, scale=True, scope='bn', zero_scale_init=True)
                d[name + '/conv_2' + '/gn'] = x

            x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            if not self.erase_relu:
                x = self.relu(x, name='relu')
            d[name] = x

        return x


class ResNet50(ResNet):
    pass


class ResNet101(ResNet):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.res_units = [None, 3, 4, 23, 3]


class ResNet50OS16(ResNet):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.dilations = [None, 1, 1, 1, 2]


class ResNet101OS16(ResNet):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.res_units = [None, 3, 4, 23, 3]
        self.dilations = [None, 1, 1, 1, 2]
