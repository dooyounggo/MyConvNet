import tensorflow as tf
from segnet import SegNet


class UNet(SegNet):
    def _init_params(self):
        self.channels = [64, 128, 256, 512]
        self.is_bn = True

    def _build_model(self, **kwargs):
        d = dict()

        x = self.X

        encoder_channels = self.channels
        for i, c in enumerate(encoder_channels):
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                x = self.encoder(x, c, is_bn=self.is_bn)
                d['block_{}'.format(self._curr_block)] = x
                if i < len(encoder_channels) - 1:
                    x = self.max_pool(x, [2, 2], [2, 2], padding='SAME')
            self._curr_block += 1

        return d

    def _build_model_seg(self, d_backbone, **kwargs):
        d = dict()

        x = d_backbone['block_{}'.format(self._curr_block - 1)]
        decoder_channels = self.channels[-2::-1]
        encoder_block = self._curr_block - 2
        for c in decoder_channels:
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                x = self.upsampling_2d_layer(x, scale=2)
                x = self.decoder(x, d_backbone['block_{}'.format(encoder_block)], c)
                d['block_{}'.format(self._curr_block)] = x
            self._curr_block += 1
            encoder_block -= 1

        self._num_blocks = self._curr_block
        self._curr_block = None
        with tf.variable_scope('logits'):
            x = self.conv_layer(x, 1, 1, self.num_classes, padding='SAME')
            if self.channel_first:
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            d['logits'] = x
            d['pred'] = tf.nn.softmax(x)

        return d

    def encoder(self, x, channels, is_bn=False):
        with tf.variable_scope('conv_0'):
            x = self.conv_layer(x, [3, 3], [1, 1], out_channels=channels, padding='SAME', biased=not is_bn)
            if self.is_bn:
                x = self.batch_norm(x)
            x = self.relu(x)

        with tf.variable_scope('conv_1'):
            x = self.conv_layer(x, [3, 3], [1, 1], out_channels=channels, padding='SAME', biased=not is_bn)
            if self.is_bn:
                x = self.batch_norm(x)
            x = self.relu(x)

        return x

    def decoder(self, x, skip, channels, is_bn=False):
        x = tf.concat([x, skip], axis=1 if self.channel_first else -1)
        with tf.variable_scope('conv_0'):
            x = self.conv_layer(x, [3, 3], [1, 1], out_channels=channels, padding='SAME', biased=not is_bn)
            if self.is_bn:
                x = self.batch_norm(x)
            x = self.relu(x)

        with tf.variable_scope('conv_1'):
            x = self.conv_layer(x, [3, 3], [1, 1], out_channels=channels, padding='SAME', biased=not is_bn)
            if self.is_bn:
                x = self.batch_norm(x)
            x = self.relu(x)

        return x
