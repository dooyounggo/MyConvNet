import tensorflow as tf
from models.resnet import ResNetBot


class ResCBAMNet(ResNetBot):    # Residual networks with Convolutional Block Attention Modules (Based on ResNet-D)
    def _init_params(self):
        self.channels = [64, 256, 512, 1024, 2048]
        self.kernels = [7, 3, 3, 3, 3]
        self.strides = [2, 1, 2, 2, 2]
        self.res_units = [3, 4, 6, 3]

        self.cam_ratio = 4
        self.sam_kernel = 7

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

            if stride[0] > 1 or stride[1] > 1:
                skip = self.avg_pool(x, stride, stride, 'SAME')
            else:
                skip = x
            if in_channels != out_channels:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(skip, 1, 1, out_channels, padding='SAME')
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
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train,
                                    zero_scale_init=True, scope='bn')
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

            x = skip + x*survival
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


class ResCBAMNet50(ResCBAMNet):
    def _init_params(self):
        super()._init_params()
        self.res_units = [3, 4, 6, 3]


class ResCBAMNet101(ResCBAMNet):
    def _init_params(self):
        super()._init_params()
        self.res_units = [3, 4, 23, 3]
