"""
Global Convolutional Network
https://arxiv.org/abs/1703.02719
"""
import tensorflow as tf
from segmentation.segnet import SegNet
from models.resnet_v2_d_cbam import ResNetCBAM50 as ResNet
from models.efficientnet import EfficientNetB3 as EffNet


class GCN(SegNet, ResNet):     # Global Convolutional Networks
    def _init_params(self, **kwargs):
        ResNet._init_params(self, **kwargs)
        self.block_activations = False
        self.max_conv_channels = 64
        self.conv_channels = [4*self.num_classes, 3*self.num_classes, 2*self.num_classes, self.num_classes]
        self.conv_kernels = [25, 25, 25, 25]
        self.conv_units = [1, 1, 1, 1]
        self.deconv_method = 'UPSAMPLING'    # Upsampling: bilinear up-sampling, conv: transposed convolution

    def _build_model_seg(self, d_backbone):
        d = dict()

        cc = self.conv_channels
        ck = self.conv_kernels
        conv_units = self.conv_units
        deconv_method = self.deconv_method

        len_c = len(cc)
        len_k = len(ck)
        len_r = len(conv_units)
        self._num_conv_blocks = min([len_c, len_k, len_r])

        assert self._curr_block == self._num_conv_blocks, 'The numbers of res and conv blocks must match'
        self._input_block = self._curr_block

        self._curr_block += 1
        x = d_backbone['block_{}'.format(self._input_block)]
        self._input_block -= 1
        for j in range(conv_units[0]):
            x = self._conv_unit(x, ck[0], cc[0], d, name='block_{}/conv_{}'.format(self._curr_block, j))
            d['block_{}'.format(self._curr_block) + '/conv_{}'.format(j)] = x
        x = self._br_unit(x, d, name='block_{}/br'.format(self._curr_block))
        d['block_{}'.format(self._curr_block) + '/br'] = x
        x = self._deconv_unit(x, d, scale=2, method=deconv_method,
                              name='block_{}/deconv'.format(self._curr_block))
        d['block_{}'.format(self._curr_block) + '/deconv'] = x

        for i in range(1, self._num_conv_blocks):
            self._curr_block += 1
            x = d_backbone['block_{}'.format(self._input_block)]
            self._input_block -= 1
            for j in range(conv_units[i]):
                x = self._conv_unit(x, ck[i], cc[i], d, name='block_{}/conv_{}'.format(self._curr_block, j))
                d['block_{}'.format(self._curr_block) + '/conv'] = x
            x = self._br_unit(x, d, name='block_{}/br_0'.format(self._curr_block))
            d['block_{}'.format(self._curr_block) + '/br_0'] = x
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                prev_block = d['block_{}'.format(self._curr_block - 1) + '/deconv']
                num_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
                num_prev_channels = prev_block.get_shape()[1] if self.channel_first else prev_block.get_shape()[-1]
                if num_channels != num_prev_channels:
                    prev_block = self.conv_layer(prev_block, 1, 1, num_channels, padding='SAME')
                x = x + prev_block
            x = self._br_unit(x, d, name='block_{}/br_1'.format(self._curr_block))
            d['block_{}'.format(self._curr_block) + '/br_1'] = x
            x = self._deconv_unit(x, d, scale=2, method=deconv_method,
                                  name='block_{}/deconv'.format(self._curr_block))
            d['block_{}'.format(self._curr_block) + '/deconv'] = x

        self._curr_block += 1
        x = self._br_unit(x, d, name='block_{}/br_0'.format(self._curr_block))
        d['block_{}'.format(self._curr_block) + '/br_0'] = x
        x = self._deconv_unit(x, d, scale=2, method=deconv_method, name='block_{}/deconv'.format(self._curr_block))
        d['block_{}'.format(self._curr_block) + '/deconv'] = x
        x = self._br_unit(x, d, name='block_{}/br_1'.format(self._curr_block))
        d['block_{}'.format(self._curr_block) + '/br_1'] = x

        self._curr_block = None
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('logits'):
                x = self.conv_layer(x, 1, 1, self.num_classes)
                if self.channel_first:
                    x = tf.transpose(x, perm=[0, 2, 3, 1])
                d['logits'] = x
                d['pred'] = tf.nn.softmax(x)

        return d

    def _conv_unit(self, x, kernel, out_channels, d, name='conv_unit'):
        if not isinstance(kernel, list):
            kernel = [kernel, kernel]
        elif len(kernel) == 1:
            kernel = [kernel[0], kernel[0]]
        out_channels = min([self.max_conv_channels, out_channels])

        with tf.variable_scope(name):
            with tf.variable_scope('conv_0'):
                x0 = self.conv_layer(x, [kernel[0], 1], 1, out_channels)
                d[name + '/conv_0'] = x0
            with tf.variable_scope('conv_1'):
                x1 = self.conv_layer(x, [1, kernel[1]], 1, out_channels)
                d[name + '/conv_1'] = x1
            with tf.variable_scope('conv_2'):
                x0 = self.conv_layer(x0, [1, kernel[1]], 1, out_channels)
                d[name + '/conv_2'] = x0
            with tf.variable_scope('conv_3'):
                x1 = self.conv_layer(x1, [kernel[0], 1], 1, out_channels)
                d[name + '/conv_3'] = x1
            x = x0 + x1
            print(name + '.shape', x.get_shape().as_list())

        return x

    def _br_unit(self, x, d, name='boundary_refinement'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]

        with tf.variable_scope(name):
            skip = x
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 3, 1, in_channels)
                x = self.relu(x)
                d[name + '/conv_0'] = x
            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, 3, 1, in_channels)
                d[name + '/conv_1'] = x
            x = x + skip

        return x

    def _deconv_unit(self, x, d, scale=2, method='upsampling', kernel=3, out_channels=None, name='upsampling'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if out_channels is None:
            out_channels = in_channels

        with tf.variable_scope(name):
            if method.lower() == 'upsampling':
                x = self.upsampling_2d_layer(x, scale, name='upsampling')
            else:
                x = self.transposed_conv_layer(x, kernel, scale, out_channels)
        print(name + '.shape', x.get_shape().as_list())

        return x


class SCN(GCN, EffNet):  # Separable Convolution Networks: GCN with separable convolution and efficientnet backbone
    def _init_params(self, **kwargs):
        EffNet._init_params(self, **kwargs)
        self.max_conv_channels = 128
        self.conv_channels = [self.num_classes, self.num_classes, self.num_classes, self.num_classes]
        self.conv_kernels = [5, 9, 13, 17]
        self.conv_units = [1, 1, 1, 1]
        self.deconv_method = 'UPSAMPLING'  # Upsampling: bilinear up-sampling, conv: transposed convolution

    def _build_model(self):
        d = EffNet._build_model(self)
        return d

    def _mb_conv_unit(self, x, kernel, stride, out_channels, multiplier, d, drop_rate=0.0, name='res_unit'):
        x = EffNet._mb_conv_unit(self, x, kernel, stride, out_channels, multiplier, d, drop_rate=drop_rate, name=name)
        return x

    def _conv_unit(self, x, kernel, out_channels, d, name='conv_unit'):
        in_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
        if not isinstance(kernel, list):
            kernel = [kernel, kernel]
        elif len(kernel) == 1:
            kernel = [kernel[0], kernel[0]]

        with tf.variable_scope(name):
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=True, depthwise=False)
                d[name + '/conv_0'] = x
            with tf.variable_scope('conv_1'):
                x = self.conv_layer(x, kernel, 1, out_channels, padding='SAME', biased=True, depthwise=True)
                d[name + '/conv_1'] = x
            with tf.variable_scope('conv_2'):
                x = self.conv_layer(x, 1, 1, out_channels, padding='SAME', biased=True, depthwise=False)
                d[name + '/conv_2'] = x
            print(name + '.shape', x.get_shape().as_list())

        return x
