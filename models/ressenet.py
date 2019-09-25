import tensorflow as tf
from models.resnet import ResNetID


class ResSENet(ResNetID):   # Residual squeeze-and-excitation networks
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
                                   dtype=self.dtype)/(tf.constant(1.0, dtype=self.dtype) - drop_rate)
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
                # x = self.relu(x, name='relu')
                # d[name + '/conv_0' + '/relu'] = x
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

            se_mask = self._se_mask(x, 16, name='se_mask')
            d[name + '/se_mask'] = se_mask
            x = x*se_mask

            x = skip + x*survival
            d[name] = x

        return x

    def _se_mask(self, x, se_r, name='se_mask'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        axis = [2, 3] if self.channel_first else [1, 2]
        with tf.variable_scope(name):
            x = tf.reduce_mean(x, axis=axis)

            with tf.variable_scope('fc1'):
                x = self.fc_layer(x, in_channels//se_r)

            x = self.relu(x, name='relu')

            with tf.variable_scope('fc2'):
                x = self.fc_layer(x, in_channels)

            x = self.sigmoid(x)
            batch_size = tf.shape(x)[0]
            shape = [batch_size, in_channels, 1, 1] if self.channel_first else [batch_size, 1, 1, in_channels]
            x = tf.reshape(x, shape=shape)

        return x


class ResSENetBot(ResSENet):    # Residual squeeze-and-excitation networks with bottlenecks
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
                # x = self.relu(x, name='relu')
                # d[name + '/conv_0' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels//4, padding='SAME')
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            with tf.variable_scope('conv_1'):
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x
                x = self.conv_layer(x, kernel, stride, out_channels//4, padding='SAME')
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

            se_mask = self._se_mask(x, 16, name='se_mask')
            d[name + '/se_mask'] = se_mask
            x = x*se_mask

            x = skip + x*survival
            d[name] = x

        return x


class ResSENet18(ResSENet):
    def _init_params(self):
        super()._init_params()
        self.channels = [64, 64, 128, 256, 512]
        self.res_units = [2, 2, 2, 2]


class ResSENet50(ResSENetBot):
    def _init_params(self):
        super()._init_params()
        self.channels = [64, 256, 512, 1024, 2048]
        self.res_units = [3, 4, 6, 3]


class ResSENet101(ResSENetBot):
    def _init_params(self):
        super()._init_params()
        self.channels = [64, 256, 512, 1024, 2048]
        self.res_units = [3, 4, 23, 3]
