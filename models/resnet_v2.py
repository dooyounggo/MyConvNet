"""
ResNet-V2
https://arxiv.org/abs/1603.05027
"""
import tensorflow as tf
from models.resnet_v1_5 import ResNet


class ResNetID(ResNet):  # ResNet with identity connections (ResNet-v2) and stochastic depth
    def _build_model(self):
        d = dict()

        X_input = self.X

        channels = self.channels
        kernels = self.kernels
        strides = self.strides
        res_units = self.res_units

        len_c = len(channels)
        len_k = len(kernels)
        len_s = len(strides)
        len_r = len(res_units) + 1
        num_blocks = min([len_c, len_k, len_s, len_r])

        self._curr_block = 0
        with tf.variable_scope('block_0'):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(X_input, kernels[0], strides[0], channels[0], padding='SAME')
                print('block_0' + '/conv_0.shape', x.get_shape().as_list())
                d['block_0' + '/conv_0'] = x
                # x = self.batch_norm(x, center=True, scale=True, scope='bn')
                # d['block_0' + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d['block_0' + '/conv_0' + '/relu'] = x
                x = self.max_pool(x, 3, 2, padding='SAME')
                d['block_0' + '/conv_0' + '/maxpool'] = x
            d['block_0'] = x

        for i in range(1, num_blocks):
            self._curr_block = i
            dr = self.initial_drop_rate + (self.final_drop_rate - self.initial_drop_rate)*i/(num_blocks - 1)
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
                    x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                    d['logits' + '/bn'] = x
                    x = self.relu(x, name='relu')
                    d['logits' + '/relu'] = x

                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x
                    x = tf.nn.dropout(x, rate=self.dropout_rate_features)
                    x = self.fc_layer(x, self.num_classes)
                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)

        return d

    def _res_unit(self, x, kernel, stride, out_channels, d, drop_rate=0.0, name='res_unit'):
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
                    skip = self.conv_layer(x, 1, stride, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_0' + '/relu'] = x
                x = self.conv_layer(x, kernel, stride, out_channels, padding='SAME')
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            with tf.variable_scope('conv_1'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x
                x = self.conv_layer(x, 3, 1, out_channels, padding='SAME')
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x

            x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            d[name] = x

        return x


class ResNetBot(ResNetID):  # ResNet with bottlenecks. ResNet-50
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.channels = [64, 256, 512, 1024, 2048]
        self.res_units = [3, 4, 6, 3]

    def _res_unit(self, x, kernel, stride, out_channels, d, drop_rate=0.0, name='res_unit'):
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
                    skip = self.conv_layer(x, 1, stride, out_channels, padding='SAME')
            d[name + '/branch'] = skip

            with tf.variable_scope('conv_0'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_0' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_0' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels//4, padding='SAME', biased=False)
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            with tf.variable_scope('conv_1'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_1' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_1' + '/relu'] = x
                x = self.conv_layer(x, kernel, stride, out_channels//4, padding='SAME', biased=False)
                print(name + '/conv_1.shape', x.get_shape().as_list())
                d[name + '/conv_1'] = x

            with tf.variable_scope('conv_2'):
                x = self.batch_norm(x, shift=True, scale=True, scope='bn')
                d[name + '/conv_2' + '/bn'] = x
                x = self.relu(x, name='relu')
                d[name + '/conv_2' + '/relu'] = x
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME')
                print(name + '/conv_2.shape', x.get_shape().as_list())
                d[name + '/conv_2'] = x

            x = self.stochastic_depth(x, skip, drop_rate=drop_rate)
            d[name] = x

        return x


class ResNet18(ResNetID):
    pass


class ResNet34(ResNetID):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.res_units = [3, 4, 6, 3]


class ResNet50(ResNetBot):
    pass


class ResNet101(ResNetBot):
    def _init_params(self, **kwargs):
        super()._init_params(**kwargs)
        self.res_units = [3, 4, 23, 3]
