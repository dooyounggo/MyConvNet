import tensorflow as tf
from segnet import SegNet
from models.resnet_v1_5_dilated import ResNet50OS16 as ResNet


class DeepLabV3PlusResNet(SegNet, ResNet):
    def _init_params(self):
        ResNet._init_params(self)
        self.feature_blocks = [4, 1]
        self.feature_channels = [256, 48]
        self.feature_gradients = [None, True]
        self.drop_rate_multipliers = [1.0, 0.0]

        self.conv_kernels = [None, 3]
        self.groups = [16, 8]  # Group normalization number of groups

        self.aspp_dilations = [6, 12, 18]

    def _build_model_seg(self, d_backbone, **kwargs):
        d = dict()

        blocks = self.feature_blocks
        feature_channels = self.feature_channels
        gradients = self.feature_gradients
        drop_rate_multipliers = self.drop_rate_multipliers
        kernels = self.conv_kernels
        groups = self.groups

        self._num_decoder_blocks = min([len(blocks), len(feature_channels), len(gradients),
                                        len(drop_rate_multipliers), len(kernels), len(groups)])
        self._num_blocks = self._curr_block + self._num_decoder_blocks + 1

        self._curr_block += 1
        feat = d_backbone['block_{}'.format(blocks[0])]
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            feat = self.aspp_unit(feat, feature_channels[0], self.aspp_dilations, groups[0])
            print('block_{}'.format(self._curr_block) + '/feature.shape', feat.get_shape().as_list())
            x = feat
        d['block_{}'.format(self._curr_block)] = x

        for i in range(1, self._num_decoder_blocks):
            self._curr_block += 1
            if gradients[i]:
                feat = d_backbone['block_{}'.format(blocks[i])]
            else:
                feat = tf.stop_gradient(d_backbone['block_{}'.format(blocks[i])])
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                with tf.variable_scope('features'):
                    feat = tf.nn.dropout(feat, rate=self.dropout_rate_features*drop_rate_multipliers[i])
                    feat = self.conv_layer(feat, 1, 1, feature_channels[i], biased=False, ws=True)
                    feat = self.group_norm(feat, num_groups=groups[i], shift=True, scale=True, scope='norm')
                print('block_{}'.format(self._curr_block) + '/feature.shape', feat.get_shape().as_list())

                size = feat.get_shape()[2:4] if self.channel_first else feat.get_shape()[1:3]
                x = self.upsampling_2d_layer(x, out_shape=size, align_corners=True)
            x = self.decoder_unit(x, feat, kernels[i], groups[0], d, name='block_{}/decoder'.format(self._curr_block))
            d['block_{}'.format(self._curr_block)] = x

        self._curr_block = None
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            with tf.variable_scope('logits'):
                x = self.conv_layer(x, 1, 1, self.num_classes)
                x = self.upsampling_2d_layer(x, out_shape=self.input_size[0:2], align_corners=True)
                if self.channel_first:
                    x = tf.transpose(x, perm=[0, 2, 3, 1])
                d['logits'] = x
                d['pred'] = tf.nn.softmax(x)

        return d

    def aspp_unit(self, x, channels, dilations, group, name='aspp'):
        axis = 1 if self.channel_first else -1
        with tf.variable_scope(name):
            ys = []
            with tf.variable_scope('conv_0'):
                y = self.conv_layer(x, 1, 1, channels, padding='SAME', biased=False, depthwise=False, ws=True)
                y = self.group_norm(y, num_groups=group, shift=True, scale=True, scope='norm')
                ys.append(y)
            for i in range(len(dilations)):
                with tf.variable_scope('conv_{}'.format(i + 1)):
                    y = self.conv_layer(x, 3, 1, channels, padding='SAME', biased=False, depthwise=False,
                                        dilation=dilations[i], ws=True)
                    y = self.group_norm(y, num_groups=group, shift=True, scale=True, scope='norm')
                    ys.append(y)
            with tf.variable_scope('conv_pool'):
                pool_axis = [2, 3] if self.channel_first else [1, 2]
                shape = x.get_shape()[2:4] if self.channel_first else x.get_shape()[1:3]
                y = tf.reduce_mean(x, axis=pool_axis, keepdims=True)
                y = self.conv_layer(y, 1, 1, channels, padding='SAME', biased=False, depthwise=False,
                                    dilation=dilations[i], ws=True)
                y = self.group_norm(y, num_groups=group, shift=True, scale=True, scope='norm')
                y = self.upsampling_2d_layer(y, out_shape=shape)
                ys.append(y)
            with tf.variable_scope('conv_out'):
                x = tf.concat(ys, axis=axis)
                x = self.conv_layer(x, 1, 1, channels, padding='SAME', biased=False, depthwise=False, ws=True)
                x = self.group_norm(x, num_groups=group, shift=True, scale=True, scope='norm')

        return x

    def decoder_unit(self, x, feature, kernel, group, d, name='decoder'):
        with tf.variable_scope(name):
            out_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
            axis = 1 if self.channel_first else -1
            x = tf.concat([x, feature], axis=axis)
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, kernel, 1, out_channels, padding='SAME', biased=False,
                                    depthwise=False, ws=True)
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x
                x = self.group_norm(x, num_groups=group, shift=True, scale=True, scope='norm')
                d[name + '/conv_0' + '/norm'] = x

            d[name] = x

        return x
