"""
DeepLabV3+
https://arxiv.org/abs/1802.02611
"""
import tensorflow.compat.v1 as tf
from segmentation.segnet import SegNet
from models.resnet_v1_5_dilated import ResNet101OS16 as ResNet


class DeepLabV3PlusResNet(SegNet, ResNet):  # No BN model
    def _init_params(self, **kwargs):
        ResNet._init_params(self, **kwargs)
        self.feature_blocks = [4, 1]
        self.feature_channels = [256, 48]
        self.feature_gradients = [None, True]
        self.drop_rate_multipliers = [1.0, 0.0]

        self.conv_kernels = [None, 3]

        self.aspp_dilations = [6, 12, 18]
        self.aspp_level_feature = False

    def _build_model(self):
        return ResNet._build_model(self)

    def _build_model_seg(self, d_backbone):
        d = dict()

        blocks = self.feature_blocks
        feature_channels = self.feature_channels
        gradients = self.feature_gradients
        drop_rate_multipliers = self.drop_rate_multipliers
        kernels = self.conv_kernels

        self._num_decoder_blocks = min([len(blocks), len(feature_channels), len(gradients),
                                        len(drop_rate_multipliers), len(kernels)])

        self._curr_block += 1
        feat = d_backbone['block_{}'.format(blocks[0])]
        with tf.variable_scope('block_{}'.format(self._curr_block)):
            feat = self.aspp_unit(feat, feature_channels[0], self.aspp_dilations, level_feature=self.aspp_level_feature)
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
                    feat = self.conv_layer(feat, 1, 1, feature_channels[i], biased=False)
                    feat = self.normalization(feat, norm_type=self.norm_type, norm_param=self.norm_param)
                print('block_{}'.format(self._curr_block) + '/feature.shape', feat.get_shape().as_list())

                size = feat.get_shape()[2:4] if self.channel_first else feat.get_shape()[1:3]
                x = self.upsampling_2d_layer(x, out_shape=size, align_corners=True)
            x = self.decoder_unit(x, feat, kernels[i], d, name='block_{}/decoder'.format(self._curr_block))
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

    def aspp_unit(self, x, channels, dilations, level_feature=False, name='aspp'):
        axis = 1 if self.channel_first else -1
        with tf.variable_scope(name):
            ys = []
            with tf.variable_scope('conv_0'):
                y = self.conv_layer(x, 1, 1, channels, padding='SAME', biased=False, depthwise=False)
                y = self.normalization(y, norm_type=self.norm_type, norm_param=self.norm_param)
                ys.append(y)
            for i in range(len(dilations)):
                with tf.variable_scope('conv_{}'.format(i + 1)):
                    y = self.conv_layer(x, 3, 1, channels, padding='SAME', biased=False, depthwise=False,
                                        dilation=dilations[i])
                    y = self.normalization(y, norm_type=self.norm_type, norm_param=self.norm_param)
                    ys.append(y)
            if level_feature:
                with tf.variable_scope('conv_pool'):
                    pool_axis = [2, 3] if self.channel_first else [1, 2]
                    shape = x.get_shape()[2:4] if self.channel_first else x.get_shape()[1:3]
                    y = tf.reduce_mean(x, axis=pool_axis, keepdims=True)
                    y = self.conv_layer(y, 1, 1, channels, padding='SAME', biased=False, depthwise=False,
                                        dilation=dilations[i])
                    y = self.normalization(y, norm_type=self.norm_type, norm_param=self.norm_param)
                    y = self.upsampling_2d_layer(y, out_shape=shape)
                    ys.append(y)
            with tf.variable_scope('conv_out'):
                x = tf.concat(ys, axis=axis)
                x = self.conv_layer(x, 1, 1, channels, padding='SAME', biased=False, depthwise=False)
                x = self.normalization(x, norm_type=self.norm_type, norm_param=self.norm_param)

        return x

    def decoder_unit(self, x, feature, kernel, d, name='decoder'):
        with tf.variable_scope(name):
            out_channels = x.get_shape()[1] if self.channel_first else x.get_shape()[-1]
            axis = 1 if self.channel_first else -1
            x = tf.concat([x, feature], axis=axis)
            with tf.variable_scope('conv_0'):
                x = self.conv_layer(x, kernel, 1, out_channels, padding='SAME', biased=False, depthwise=False)
                x = self.normalization(x, norm_type=self.norm_type, norm_param=self.norm_param)
                print(name + '/conv_0.shape', x.get_shape().as_list())
                d[name + '/conv_0'] = x

            d[name] = x

        return x
