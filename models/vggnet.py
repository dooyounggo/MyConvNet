"""
VGGNet
https://arxiv.org/abs/1409.1556
"""
import tensorflow.compat.v1 as tf
from convnet import ConvNet

VGG_MEAN = [123.68, 116.78, 103.94]  # RGB mean


class VGGNet(ConvNet):
    def _init_params(self, **kwargs):
        self.num_layers = 16  # 16 or 19

    def _build_model(self):
        d = dict()

        assert self.num_layers == 16 or self.num_layers == 19, 'Number of layers must be either 16 or 19.'
        if self.dtype is not tf.float32:
            X_input = tf.cast(self.X, dtype=tf.float32)
        else:
            X_input = self.X
        x = (X_input/self.scale_factor + self.image_mean)*255.0 - tf.constant(VGG_MEAN, dtype=tf.float32)
        if self.dtype is not tf.float32:
            x = tf.cast(self.X, dtype=self.dtype)

        self._curr_block = 0
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.conv_relu(x, 64, name='conv_0')
            d['conv1_1'] = x  # VGG naming convention

            x = self.conv_relu(x, 64, name='conv_1')
            d['conv1_2'] = x

            x = self.max_pool(x, 2, 2)
        d[f'block_{self._curr_block}'] = x

        self._curr_block = 1
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.conv_relu(x, 128, name='conv_0')
            d['conv2_1'] = x

            x = self.conv_relu(x, 128, name='conv_1')
            d['conv2_2'] = x

            x = self.max_pool(x, 2, 2)
        d[f'block_{self._curr_block}'] = x

        self._curr_block = 2
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.conv_relu(x, 256, name='conv_0')
            d['conv3_1'] = x

            x = self.conv_relu(x, 256, name='conv_1')
            d['conv3_2'] = x

            x = self.conv_relu(x, 256, name='conv_2')
            d['conv3_3'] = x

            if self.num_layers == 19:
                x = self.conv_relu(x, 256, name='conv_3')
                d['conv3_4'] = x

            x = self.max_pool(x, 2, 2)
        d[f'block_{self._curr_block}'] = x

        self._curr_block = 3
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.conv_relu(x, 512, name='conv_0')
            d['conv4_1'] = x

            x = self.conv_relu(x, 512, name='conv_1')
            d['conv4_2'] = x

            x = self.conv_relu(x, 512, name='conv_2')
            d['conv4_3'] = x

            if self.num_layers == 19:
                x = self.conv_relu(x, 512, name='conv_3')
                d['conv4_4'] = x

            x = self.max_pool(x, 2, 2)
        d[f'block_{self._curr_block}'] = x

        self._curr_block = 4
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.conv_relu(x, 512, name='conv_0')
            d['conv5_1'] = x

            x = self.conv_relu(x, 512, name='conv_1')
            d['conv5_2'] = x

            x = self.conv_relu(x, 512, name='conv_2')
            d['conv5_3'] = x

            if self.num_layers == 19:
                x = self.conv_relu(x, 512, name='conv_3')
                d['conv5_4'] = x

            x = self.max_pool(x, 2, 2)
        d[f'block_{self._curr_block}'] = x

        if not self.backbone_only:
            self._curr_block = None
            with tf.variable_scope(f'block_{self._curr_block}'):
                assert self.input_size[0] == 224 and self.input_size[1] == 224, 'Input shape must be (224, 224, 3)'
                x = self.conv_layer(x, 7, stride=1, out_channels=4096, padding='VALID',
                                    biased=True, scope='fc_0', verbose=True)
                x = self.relu(x)
                d['fc6'] = x

                x = self.conv_layer(x, 1, stride=1, out_channels=4096, padding='SAME',
                                    biased=True, scope='fc_1', verbose=True)
                x = self.relu(x)
                d['fc7'] = x

                x = self.conv_layer(x, 1, stride=1, out_channels=self.num_classes, padding='SAME',
                                    biased=True, scope='fc_2', verbose=True)
                d['fc8'] = x

                x = tf.reshape(x, shape=[-1, self.num_classes])
                d['logits'] = x
                d['pred'] = tf.nn.softmax(x)

        return d

    def conv_relu(self, x, channels, name, verbose=True):
        x = self.conv_layer(x, 3, stride=1, out_channels=channels, padding='SAME',
                            biased=True, scope=name, verbose=verbose)
        x = self.relu(x)
        return x


class VGG16(VGGNet):
    def _init_params(self, **kwargs):
        self.num_layers = 16


class VGG19(VGGNet):
    def _init_params(self, **kwargs):
        self.num_layers = 19
