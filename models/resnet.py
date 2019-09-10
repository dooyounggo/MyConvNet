import tensorflow as tf
from convnet import ConvNet


class ResNet(ConvNet):      # Base model. ResNet-18
    def _init_params(self):
        self.channels = [64, 64, 128, 256, 512]
        self.kernels = [7, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 2]
        self.res_units = [2, 2, 2, 2]    # Number of residual units starting from the 1st block

    def _build_model(self, **kwargs):
        d = dict()

        initial_drop_rate = kwargs.get('initial_drop_rate', 0.0)
        final_drop_rate = kwargs.get('final_drop_rate', 0.0)

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
                x = self.conv_layer(X_input, kernels[0], strides[0], channels[0], padding='SAME')
                print('block_0' + '/conv_0.shape', x.get_shape().as_list())
                d['block_0' + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d['block_0' + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_0' + '/relu'] = x
                x = self.max_pool(x, 3, 2, padding='SAME')
                d['block_0' + '/conv_0' + '/maxpool'] = x
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
                x = self._res_unit(x, kernels[i], s, channels[i], d, drop_rate=dr, name='block_{}/res_{}'.format(i, j))
            d['block_{}'.format(i)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['avgpool'] = x
                    x = tf.nn.dropout(x, rate=self.dropout_rate_logits)
                    x = self.fc_layer(x, self.num_classes)
                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        else:
            if len(stride) == 1:
                stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            if in_channels == out_channels:
                if stride[0] > 1 | stride[1] > 1:
                    skip = tf.nn.max_pool(x, [1, stride[0], stride[1], 1], [1, stride[0], stride[1], 1], 'VALID')
                else:
                    skip = x
            else:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(x, 1, stride, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, kernel, stride, out_channels, padding='SAME')
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_0' + '/relu'] = x

            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, 3, 1, out_channels, padding='SAME')
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = skip + x
                d[name + '/skip'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x

            return x


class ResNetID(ResNet):     # ResNet with identity connections (ResNet-v2) and stochastic depth
    def _build_model(self, **kwargs):
        d = dict()

        initial_drop_rate = kwargs.get('initial_drop_rate', 0.0)
        final_drop_rate = kwargs.get('final_drop_rate', 0.0)

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
                x = self.conv_layer(X_input, kernels[0], strides[0], channels[0], padding='SAME')
                print('block_0' + '/conv_0.shape', x.get_shape().as_list())
                d['block_0' + '/conv_0'] = x
                # x = self.batch_norm(x, center=True, scale=True, is_training=self.is_train, scope='bn')
                # d['block_0' + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_0' + '/relu'] = x
                x = self.max_pool(x, 3, 2, padding='SAME')
                d['block_0' + '/conv_0' + '/maxpool'] = x
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
                x = self._res_unit(x, kernels[i], s, channels[i], d, drop_rate=dr, name='block_{}/res_{}'.format(i, j))
            d['block_{}'.format(self._curr_block)] = x

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('logits'):
                    x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                    d['logits' + '/bn'] = x
                    x = self.relu(x, name='relu')
                    d['logits' + '/relu'] = x

                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['avgpool'] = x
                    x = tf.nn.dropout(x, rate=self.dropout_rate_logits)
                    x = self.fc_layer(x, self.num_classes)
                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        else:
            if len(stride) == 1:
                stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            with tf.variable_scope('drop'):
                drop_rate = tf.constant(drop_rate, dtype=self.dtype, name='drop_rate')
                drop_rate = tf.cond(self.is_train, lambda: drop_rate, lambda: tf.constant(0.0, dtype=self.dtype))
                survival = tf.cast(tf.math.greater_equal(tf.random.uniform([1], dtype=self.dtype), drop_rate),
                                   dtype=self.dtype) / (tf.constant(1.0, dtype=self.dtype) - drop_rate)
            if in_channels == out_channels:
                if stride[0] > 1 or stride[1] > 1:
                    skip = tf.nn.max_pool(x, [1, stride[0], stride[1], 1], [1, stride[0], stride[1], 1], 'VALID')
                else:
                    skip = x
            else:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(x, 1, stride, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_0' + '/relu'] = x
                x = self.conv_layer(x, kernel, stride, out_channels, padding='SAME')
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            with tf.variable_scope('conv_1'):
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x
                x = self.conv_layer(x, 3, 1, out_channels, padding='SAME')
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x

            x = skip + x*survival
            d[name] = x

        return x


class ResNetBot(ResNetID):     # ResNet with bottlenecks. ResNet-50
    def _init_params(self):
        self.channels = [64, 256, 512, 1024, 2048]
        self.kernels = [7, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 2]
        self.res_units = [3, 4, 6, 3]

    def _res_unit(self, x, kernel, stride, out_channels, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        elif len(stride) == 1:
            stride = [stride[0], stride[0]]

        with tf.variable_scope(name):
            with tf.variable_scope('drop'):
                drop_rate = tf.constant(drop_rate, dtype=self.dtype, name='drop_rate')
                drop_rate = tf.cond(self.is_train, lambda: drop_rate, lambda: tf.constant(0.0, dtype=self.dtype))
                survival = tf.cast(tf.math.greater_equal(tf.random.uniform([1], dtype=self.dtype), drop_rate),
                                   dtype=self.dtype) / (tf.constant(1.0, dtype=self.dtype) - drop_rate)
            if in_channels == out_channels:
                if stride[0] > 1 or stride[1] > 1:
                    skip = tf.nn.max_pool(x, [1, stride[0], stride[1], 1], [1, stride[0], stride[1], 1], 'VALID')
                else:
                    skip = x
            else:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(x, 1, stride, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_0' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels//4, padding='SAME')
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            with tf.variable_scope('conv_1'):
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x
                x = self.conv_layer(x, kernel, stride, out_channels // 4, padding='SAME')
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x

            with tf.variable_scope('conv_2'):
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_2' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_2' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME')
                print(name + '/conv_2.shape', x.get_shape().as_list())
                d[name + '/conv_2'] = x

            x = skip + x*survival
            d[name] = x

        return x


class ResNet18(ResNetID):
    pass


class ResNet50(ResNetBot):
    pass


class ResNet101(ResNetBot):
    def _init_params(self):
        super()._init_params()
        self.res_units = [3, 4, 23, 3]
