import tensorflow as tf
from models.rescbamnet import ResCBAMNet


class ResSepNet(ResCBAMNet):  # Based on EfficientNet + CBAM
    def _init_params(self):
        self.channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        self.kernels = [3, 3, 3, 5, 3, 5, 5, 3]
        self.strides = [2, 1, 2, 2, 2, 1, 2, 1]
        self.res_units = [1, 2, 2, 3, 3, 4, 1]
        self.multipliers = [1, 2, 3, 4, 5, 6, 4]

        self.cam_ratio = 8
        self.sam_kernel = 7

        self.pool_type = 'MAX'  # 'MAX', 'AVG', 'CONV'

        self.logits_power = 0.5

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
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d['block_0' + '/conv_0' + '/bn'] = x
                # x = self.swish(x, name='swish')
                # d['block_0' + '/conv_0' + '/swish'] = x
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
                        x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                        d['logits' + 'conv_0' + '/bn'] = x
                        x = self.relu(x, name='relu')
                        d['logits' + 'conv_0' + '/relu'] = x

                    axis = [2, 3] if self.channel_first else [1, 2]
                    avgpool = tf.reduce_mean(x, axis=axis)
                    d['logits' + 'avgpool'] = avgpool
                    maxpool = tf.reduce_max(x, axis=axis)
                    d['logits' + 'maxpool'] = maxpool

                    eps = tf.constant(1e-4, dtype=self.dtype)  # Prevent too large gradient values resulting from sqrt
                    x = tf.math.pow(avgpool*maxpool + eps, self.logits_power)

                    x = tf.nn.dropout(x, rate=self.dropout_rate_logits)
                    x = self.fc_layer(x, self.num_classes)

                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, multipliers, d, drop_rate=0.0, name='res_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(kernel, (list, tuple)):
            kernel = [kernel, kernel]
        elif len(kernel) == 1:
            kernel = [kernel[0], kernel[0]]
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

            if stride[0] > 1 or stride[1] > 1:
                skip = self.avg_pool(x, stride, stride, 'SAME')
            else:
                skip = x
            if in_channels != out_channels:
                with tf.variable_scope('conv_skip'):
                    skip = self.conv_layer(skip, 1, 1, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=False, depthwise=False)
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                # x = self.swish(x, name='swish')
                # d[name + '/conv_0' + '/swish'] = x

            with tf.variable_scope('conv_1'):
                if self.pool_type.lower() == 'max':
                    pool = self.max_pool(x, kernel, stride, padding='SAME')
                elif self.pool_type.lower() == 'avg':
                    pool = self.avg_pool(x, kernel, stride, padding='SAME')
                elif self.pool_type.lower() == 'conv':
                    with tf.variable_scope('conv_pool'):
                        pool = self.conv_layer(x, kernel, stride, out_channels,
                                               padding='SAME', biased=False, depthwise=True)
                else:
                    pool = self.avg_pool(tf.zeros_like(x, dtype=self.dtype), kernel, stride, padding='SAME')

                convs = [pool]
                for i in range(multipliers):
                    with tf.variable_scope('conv_{}'.format(i)):
                        c = self.conv_layer(x, kernel, stride, out_channels,
                                            padding='SAME', biased=False, depthwise=True)
                        c = self.batch_norm(c, shift=True, scale=True, is_training=self.is_train, scope='bn')
                        c = self.swish(c, name='swish')
                        convs.append(c)
                x = tf.add_n(convs)
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x

            channel_mask = self._channel_mask(x, self.cam_ratio, name='channel_mask')
            d[name + '/channel_mask'] = channel_mask
            x = x*channel_mask

            spatial_mask = self._spatial_mask(x, self.sam_kernel, name='spatial_mask')
            d[name + '/spatial_mask'] = spatial_mask
            x = x*spatial_mask

            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=False, depthwise=False)
                print(name + '/conv_2.shape', x.get_shape().as_list())
                d[name + '/conv_2'] = x
                x = self.batch_norm(x, shift=True, scale=True, is_training=self.is_train, scope='bn')
                d[name + '/conv_2' + '/bn'] = x
                # x = self.swish(x, name='swish')
                # d[name + '/conv_2' + '/swish'] = x

            x = skip + x*survival
            d[name] = x

        return x


class ResSepNetS(ResSepNet):  # B0
    def _init_params(self):
        super()._init_params()


class ResSepNetM(ResSepNet):  # B4
    def _init_params(self):
        super()._init_params()
        self.channels = [48, 24, 40, 56, 112, 160, 272, 448, 1792]
        self.res_units = [2, 4, 4, 6, 6, 8, 2]


class ResSepNetL(ResSepNet):  # B7
    def _init_params(self):
        super()._init_params()
        self.channels = [64, 32, 48, 80, 160, 224, 384, 640, 2560]
        self.res_units = [4, 7, 7, 10, 10, 13, 4]
