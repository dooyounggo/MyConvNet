import tensorflow as tf
from generative.gan import GAN


class DCGAN(GAN):
    def _build_model(self, **kwargs):
        d = dict()
        x = self.X
        self._num_blocks = 4
        channels = 64
        for i in range(self.num_blocks):
            self._curr_block = i
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                if i == 0:
                    x = self.discriminator_unit(x, 5, 2, int(channels), norm_type=None, activation_type='lrelu')
                else:
                    x = self.discriminator_unit(x, 5, 2, int(channels), norm_type='bn', activation_type='lrelu')
            d['block_{}'.format(self._curr_block)] = x
            channels *= 2

        self._curr_block = None
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('logits'):
                # axis = [2, 3] if self.channel_first else [1, 2]
                # x = tf.reduce_mean(x, axis=axis)
                shape_x = x.get_shape().as_list()
                x = tf.reshape(x, [-1, shape_x[1]*shape_x[2]*shape_x[3]])
                x = tf.nn.dropout(x, rate=self.dropout_rate_features)
                x = self.fc_layer(x, 1, verbose=True)
                d['logits'] = x
                d['pred'] = tf.nn.sigmoid(x)

        return d

    def _build_model_g(self, **kwargs):
        d = dict()
        self._curr_block = self.num_blocks

        x = self.Y
        self._num_blocks_g = 5
        channels = 512
        in_size = [int(self.input_size[0]/(2**self.num_blocks)), int(self.input_size[1]/(2**self.num_blocks))]
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('fc_0'):
                x = self.fc_layer(x, in_size[0]*in_size[1]*channels, verbose=True, biased=False)
                x = tf.reshape(x, [-1, in_size[0], in_size[1], channels])
                x = self.batch_norm(x)
        channels /= 2
        for i in range(self.num_blocks_g - 1):
            self._curr_block += 1
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                if i == self.num_blocks_g - 2:
                    x = self.generator_unit(x, 5, 2, self.input_size[-1], norm_type=None, activation_type='tanh')
                else:
                    x = self.generator_unit(x, 5, 2, int(channels), norm_type='bn', activation_type='relu')
            d['block_{}'.format(self._curr_block)] = x
            channels /= 2
        d['generate'] = x

        return d

    def discriminator_unit(self, x, kernel, stride, channels, norm_type=None, activation_type='lrelu',
                           name='discriminator_unit'):
        with tf.variable_scope(name):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, kernel, stride, out_channels=channels, biased=norm_type is None, verbose=True)
                if norm_type is not None:
                    if norm_type.lower() in ['bn', 'batch', 'batch_norm', 'batch_normalization']:
                        x = self.batch_norm(x, scope=norm_type)
                    elif norm_type.lower() in ['gn', 'group', 'group_norm', 'group_normalization']:
                        x = self.group_norm(x, num_groups=32, scope=norm_type)
                    else:
                        raise(ValueError, 'Normalization type of {} is not supported'.format(norm_type))
            x = self.activation(x, activation_type=activation_type, params=0.2)

        return x

    def generator_unit(self, x, kernel, stride, channels, norm_type=None, activation_type='relu',
                       name='generator_unit'):
        with tf.variable_scope(name):
            with tf.variable_scope('conv_0'):
                x = self.transposed_conv_layer(x, kernel, stride, out_channels=channels, biased=norm_type is None,
                                               verbose=True)
                if norm_type is not None:
                    if norm_type.lower() in ['bn', 'batch', 'batch_norm', 'batch_normalization']:
                        x = self.batch_norm(x, scope=norm_type)
                    elif norm_type.lower() in ['gn', 'group', 'group_norm', 'group_normalization']:
                        x = self.group_norm(x, num_groups=32, scope=norm_type)
                    else:
                        raise (ValueError, 'Normalization type of {} is not supported'.format(norm_type))
            x = self.activation(x, activation_type=activation_type, params=0.2)

        return x
