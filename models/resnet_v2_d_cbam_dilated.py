import tensorflow as tf
from convnet import ConvNet


class ResNetCBAMDilated(ConvNet):    # ResNet with dilated convolutions
    def _init_params(self, **kwargs):
        self.channels = [64, 256, 512, 1024, 2048]
        self.kernels = [3, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 1]
        self.res_units = [None, 3, 4, 6, 3]
        self.dilations = [None, 1, 1, 1, 2]

        self.cam_ratio = 8
        self.sam_kernel = 7

        self.multi_grid = [1, 2, 4]

        self.block_activations = False

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
                x = self.conv_layer(X_input, kernels[0], strides[0], channels[0]//2, padding='SAME', biased=False)
                print('block_0' + '/conv_0.shape', x.get_shape().as_list())
                d['block_0' + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_0' + '/relu'] = x
            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernels[0], 1, channels[0]//2, padding='SAME', biased=False)
                print('block_0' + '/conv_1.shape', x.get_shape().as_list())
                d['block_0' + '/conv_1'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_1' + '/relu'] = x
            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, kernels[0], 1, channels[0], padding='SAME', biased=False)
                print('block_0' + '/conv_2.shape', x.get_shape().as_list())
                d['block_0' + '/conv_2'] = x
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d['block_0' + '/conv_2' + '/bn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_2' + '/relu'] = x
            d['block_-1'] = x
            x = self.max_pool(x, 2, 2, padding='SAME')
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
            if self.block_activations:
                with tf.variable_scope('block_{}/act'.format(self._curr_block)):
                    x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                    d['block_{}/'.format(self._curr_block) + '/bn'] = x
                    x = self.relu(x, name='relu')
                    d['block_{}/'.format(self._curr_block) + '/relu'] = x
            d['block_{}'.format(self._curr_block)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    if not self.block_activations:
                        x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                        d['logits' + '/bn'] = x
                        x = self.relu(x, name='relu')
                        d['logits' + '/relu'] = x

                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x

                    if self.feature_reduction > 1:
                        with tf.variable_scope('comp'):  # Feature compression to prevent overfitting
                            num_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
                            x = self.fc_layer(x, num_channels//self.feature_reduction)
                            x = self.relu(x, name='relu')

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
            if stride[0] > 1 or stride[1] > 1:
                skip = self.avg_pool(x, stride, stride, 'SAME')
            else:
                skip = x
            if in_channels != out_channels:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(skip, 1, 1, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                # x = self.relu(x, name='relu')
                # d[name + '/conv_0' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels//4, padding='SAME', biased=False)
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            with tf.variable_scope('conv_1'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x
                x = self.conv_layer(x, kernel, stride, out_channels//4, padding='SAME', biased=False, dilation=dilation)
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x

            with tf.variable_scope('conv_2'):
                x = self.batch_norm(x, shift=True, scale=True, zero_scale_init=True, scope='bn')
                d[name + '/conv_2' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_2' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME')
                print(name + '/conv_2.shape', x.get_shape().as_list())
                d[name + '/conv_2'] = x

            channel_mask = self._channel_mask(x, self.cam_ratio, name='channel_mask')
            d[name + '/channel_mask'] = channel_mask
            x = x*channel_mask

            spatial_mask = self._spatial_mask(x, self.sam_kernel, name='spatial_mask')
            d[name + '/spatial_mask'] = spatial_mask
            x = x*spatial_mask

            x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            d[name] = x

        return x

    def _channel_mask(self, x, reduction, name='channel_mask'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        axis = [2, 3] if self.channel_first else [1, 2]
        with tf.variable_scope(name):
            spatial_mean = tf.reduce_mean(x, axis=axis)
            spatial_max = tf.reduce_max(x, axis=axis)

            with tf.variable_scope('fc1'):
                spatial_mean = self.fc_layer(spatial_mean, in_channels//reduction)
                tf.get_variable_scope().reuse_variables()
                spatial_max = self.fc_layer(spatial_max, in_channels//reduction)

            spatial_mean = self.relu(spatial_mean, name='relu_mean')
            spatial_max = self.relu(spatial_max, name='relu_max')

            with tf.variable_scope('fc2'):
                spatial_mean = self.fc_layer(spatial_mean, in_channels)
                tf.get_variable_scope().reuse_variables()
                spatial_max = self.fc_layer(spatial_max, in_channels)

            x = self.sigmoid(spatial_mean + spatial_max)
            batch_size = tf.shape(x)[0]
            shape = [batch_size, in_channels, 1, 1] if self.channel_first else [batch_size, 1, 1, in_channels]
            x = tf.reshape(x, shape=shape)

        return x

    def _spatial_mask(self, x, kernel, name='spatial_mask'):
        with tf.variable_scope(name):
            axis = 1 if self.channel_first else -1
            channel_mean = tf.reduce_mean(x, axis=axis, keepdims=True)
            channel_max = tf.reduce_max(x, axis=axis, keepdims=True)
            x = tf.concat([channel_mean, channel_max], axis=axis)

            with tf.variable_scope('conv'):
                x = self.conv_layer(x, kernel, 1, 1, padding='SAME', biased=True)

            x = self.sigmoid(x)

        return x


class ResNetCBAM50OS16(ResNetCBAMDilated):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.strides = [2, 1, 2, 2, 1]
        self.res_units = [None, 3, 4, 6, 3]
        self.dilations = [None, 1, 1, 1, 2]


class ResNetCBAM50OS8(ResNetCBAMDilated):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.strides = [2, 1, 2, 1, 1]
        self.res_units = [None, 3, 4, 6, 3]
        self.dilations = [None, 1, 1, 2, 4]
