"""
Unprocessing Images + Demosaicing
"""
import tensorflow.compat.v1 as tf
from isp.unprocessing_demosaic import UnprocessingDemosaic, process


class UPDMNet(UnprocessingDemosaic):
    def _init_params(self, **kwargs):
        self.channels = [32, 64, 128, 256]
        self.use_bn = kwargs.get('use_bn', False)
        self.activation_type = kwargs.get('activation_type', 'lrelu')

    def _build_model(self):
        d = dict()
        X_input = self.X
        x = X_input

        self._curr_block = 0
        residuals = []
        for i, c in enumerate(self.channels):
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                x = self.conv_unit(x, c, use_bn=self.use_bn, activation_type=self.activation_type)
                if i < len(self.channels) - 1:
                    residuals.append(x)
                    x = self.max_pool(x, 2, 2, padding='SAME')
                d['block_{}'.format(self._curr_block)] = x
            self._curr_block += 1

        channel_axis = 1 if self.channel_first else -1
        for i, c in enumerate(self.channels[-2::-1]):
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                if i < len(self.channels) - 1:
                    x = self.upsampling_2d_layer(x, scale=2, upsampling_method='bilinear')
                skip = residuals.pop()
                x = tf.concat([x, skip], axis=channel_axis)
                x = self.conv_unit(x, c, use_bn=self.use_bn, activation_type=self.activation_type)
                d['block_{}'.format(self._curr_block)] = x
            self._curr_block += 1

        features = x
        self._curr_block = 'denoise'  # Denoising head
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(features, 3, 1, out_channels=4, padding='SAME', biased=True, verbose=True)
                # x = self.tanh(x)
            noisy_img = tf.gather(X_input, [0, 1, 2, 3], axis=channel_axis)
            denoised = x + noisy_img
            d['denoised'] = denoised
            if self.channel_first:
                denoised = tf.transpose(denoised, perm=[0, 2, 3, 1])
            if self.dtype is not tf.float32:
                denoised = tf.cast(denoised, dtype=tf.float32)
            denoised_rgb = process.demosaic(denoised)
            denoised_rgb.set_shape([None] + list(self.input_size))
            if self.dtype is not tf.float32:
                denoised_rgb = tf.cast(denoised_rgb, dtype=self.dtype)
            if self.channel_first:
                denoised_rgb = tf.transpose(denoised_rgb, perm=[0, 3, 1, 2])

        self._curr_block = 'demosaic'  # Demosaicing head
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            x = self.upsampling_2d_layer(features, scale=2, upsampling_method='bilinear')
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 3, 1, out_channels=self.channels[0]//2, biased=not self.use_bn, verbose=True)
                if self.use_bn:
                    x = self.batch_norm(x)
                x = self.activation(x, activation_type=self.activation_type)
            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, 3, 1, out_channels=3, biased=True, verbose=True)
                # x = self.tanh(x)
            d['pred'] = x + denoised_rgb

        return d

    def conv_unit(self, x, channels, use_bn=False, activation_type='lrelu'):
        for i in range(3):
            with tf.variable_scope(f'conv_{i}'):
                x = self.conv_layer(x, 3, 1, out_channels=channels, padding='SAME', biased=not use_bn, verbose=True)
                if use_bn:
                    x = self.batch_norm(x)
                x = self.activation(x, activation_type=activation_type)
        return x


class EDNMNet(UnprocessingDemosaic):
    def _init_params(self, **kwargs):
        self.channels = [24, 24, 40, 80]
        self.kernels = [3, 3, 5, 5]
        self.strides = [1, 2, 2, 2]
        self.conv_units = [1, 2, 3, 4]
        self.multipliers = [1, 3, 6, 6]

        self.use_bn = kwargs.get('use_bn', False)
        self.activation_type = kwargs.get('activation_type', 'lrelu')
        self.conv_initializer = tf.initializers.variance_scaling(mode='fan_out')

        self.striding_kernel_offset = kwargs.get('striding_kernel_offset', 0)
        self.striding_kernel_size = kwargs.get('striding_kernel_size', 4)

    def _build_model(self):
        d = dict()
        channel_axis = 1 if self.channel_first else -1

        X_input = self.X
        x = X_input
        bayer = tf.gather(X_input, [0, 1, 2, 3], axis=channel_axis)
        bayer_avg = bayer - 0.5
        bayer_max = bayer_avg

        self._curr_block = 0
        residuals = []
        for i, (c, k, s, n, m) in enumerate(zip(self.channels, self.kernels, self.strides,
                                                self.conv_units, self.multipliers)):
            kernel = k
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                if s > 1:
                    bayer_avg = self.avg_pool(bayer_avg, s, s)
                    bayer_max = self.max_pool(bayer_max, s, s)
                else:
                    bayer_avg = tf.identity(bayer_avg)
                    bayer_max = tf.identity(bayer_max)
                for j in range(n):
                    k = kernel
                    if j > 0:
                        s = 1
                    else:
                        if s > 1:
                            if self.striding_kernel_size is None:
                                k += self.striding_kernel_offset
                            else:
                                k = self.striding_kernel_size
                    if i == j == 0:
                        x = self.conv_unit(x, k, s, c, activation_type=self.activation_type, name=f'conv_{j}')
                    else:
                        x = self.mbconv_unit(x, k, s, c, m, d, activation_type=self.activation_type, name=f'mbconv_{j}')
                x = tf.concat([x, bayer_avg, bayer_max], axis=channel_axis)
                if i < len(self.channels) - 1:
                    residuals.append(x)
                d['block_{}'.format(self._curr_block)] = x
            self._curr_block += 1

        for c, k, s, n, m in zip(self.channels[-2::-1], self.kernels[-2::-1], self.strides[::-1],
                                 self.conv_units[-2::-1], self.multipliers[-2::-1]):
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                if s > 1:
                    x = self.upsampling_2d_layer(x, scale=s, upsampling_method='bilinear')
                skip = residuals.pop()
                x = tf.concat([x, skip], axis=channel_axis)
                for j in range(n):
                    x = self.mbconv_unit(x, k, 1, c, m, d, activation_type=self.activation_type, name=f'mbconv_{j}')
                d['block_{}'.format(self._curr_block)] = x
            self._curr_block += 1

        if self.strides[0] > 1:
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                x = self.upsampling_2d_layer(x, scale=self.strides[0], upsampling_method='bilinear')
                features = self.conv_unit(x, self.kernels[0], 1, self.channels[0],
                                          activation_type=self.activation_type, name='conv_0')
        else:
            features = x

        self._curr_block = 'denoise'  # Denoising head
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(features, 1, 1, out_channels=4, padding='SAME', biased=False, verbose=True)
                # x = self.tanh(x)
            noisy_img = bayer
            denoised = x + noisy_img
            d['denoised'] = denoised
            if self.channel_first:
                denoised = tf.transpose(denoised, perm=[0, 2, 3, 1])
            if self.dtype is not tf.float32:
                denoised = tf.cast(denoised, dtype=tf.float32)
            denoised_rgb = process.demosaic(denoised)
            denoised_rgb.set_shape([None] + list(self.input_size))
            if self.dtype is not tf.float32:
                denoised_rgb = tf.cast(denoised_rgb, dtype=self.dtype)
            if self.channel_first:
                denoised_rgb = tf.transpose(denoised_rgb, perm=[0, 3, 1, 2])

        self._curr_block = 'demosaic'  # Demosaicing head
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            x = self.upsampling_2d_layer(features, scale=2, upsampling_method='bilinear')
            x = tf.concat([x, denoised_rgb - 0.5], axis=channel_axis)
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 3, 1, out_channels=self.channels[0]//2,
                                    padding='SAME', biased=True, verbose=True)
                x = self.activation(x, activation_type=self.activation_type)
            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, 3, 1, out_channels=3, padding='SAME', biased=False, verbose=True)
                # x = self.tanh(x)
            d['pred'] = x + denoised_rgb

        return d

    def conv_unit(self, x, kernel, stride, out_channels, activation_type='lrelu', name='conv'):
        with tf.variable_scope(name):
            x = self.conv_layer(x, kernel, stride, out_channels=out_channels, padding='SAME',
                                biased=not self.use_bn, verbose=True)
            if self.use_bn:
                x = self.batch_norm(x)
            x = self.activation(x, activation_type=activation_type)
        return x

    def mbconv_unit(self, x, kernel, stride, out_channels, multiplier, d, activation_type='lrelu', name='mbconv'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        with tf.variable_scope(name):
            if stride == 1 and in_channels == out_channels:
                skip = x
            else:
                skip = None
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 1, 1, out_channels*multiplier, padding='SAME', biased=not self.use_bn,
                                    depthwise=False, weight_initializer=self.conv_initializer, verbose=True)
                d[name + '/conv_0'] = x
                if self.use_bn:
                    x = self.batch_norm(x, shift=True, scale=True, scope='norm')
                    d[name + '/conv_0' + '/norm'] = x
                x = self.activation(x, activation_type=activation_type)
                d[name + '/conv_0' + activation_type] = x

            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernel, stride, out_channels*multiplier, padding='SAME', biased=not self.use_bn,
                                    depthwise=True, weight_initializer=self.conv_initializer, verbose=True)
                d[name + '/conv_1'] = x
                if self.use_bn:
                    x = self.batch_norm(x, shift=True, scale=True, scope='norm')
                    d[name + '/conv_1' + '/norm'] = x
                x = self.activation(x, activation_type=activation_type)
                d[name + '/conv_1' + activation_type] = x

            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=not self.use_bn, depthwise=False,
                                    weight_initializer=self.conv_initializer, verbose=True)
                d[name + '/conv_2'] = x
                if self.use_bn:
                    x = self.batch_norm(x, shift=True, scale=True, zero_scale_init=skip is not None, scope='norm')
                    d[name + '/conv_2' + '/norm'] = x

            if skip is not None:
                x += skip
            d[name] = x

        return x
