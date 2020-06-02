"""
Unprocessing Images
https://arxiv.org/abs/1811.11127
"""
import tensorflow.compat.v1 as tf
from isp.unprocessing import Unprocessing


class UPINet(Unprocessing):
    def _init_params(self, **kwargs):
        self.channels = [32, 64, 128, 256, 512]
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

        self._curr_block = None
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 3, 1, out_channels=4, padding='SAME', biased=True, verbose=True)
            noisy_img = tf.gather(X_input, [0, 1, 2, 3], axis=channel_axis)
            d['pred'] = x + noisy_img

        return d

    def conv_unit(self, x, channels, use_bn=False, activation_type='lrelu'):
        for i in range(3):
            with tf.variable_scope(f'conv_{i}'):
                x = self.conv_layer(x, 3, 1, out_channels=channels, padding='SAME', biased=not use_bn, verbose=True)
                if use_bn:
                    x = self.batch_norm(x)
                x = self.activation(x, activation_type=activation_type)
        return x
