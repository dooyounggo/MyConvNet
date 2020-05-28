import tensorflow.compat.v1 as tf
from convnet import ConvNet


class InceptionV3(ConvNet):  # Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    def _init_params(self, **kwargs):
        self.has_aux_logits = kwargs.get('aux_logits', True)

    def _build_model(self):
        d = dict()

        X_input = self.X

        self._num_blocks = 7

        self._curr_block = 0
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = X_input
            x = self.conv_bn_act(x, 3, 2, out_channels=32, padding='VALID', scale=False, verbose=True, scope='conv_0')
            x = self.conv_bn_act(x, 3, 1, out_channels=32, padding='VALID', scale=False, verbose=True, scope='conv_1')
            x = self.conv_bn_act(x, 3, 1, out_channels=64, padding='SAME', scale=False, verbose=True, scope='conv_2')
            x = self.max_pool(x, 3, 2, padding='VALID')
            x = self.conv_bn_act(x, 1, 1, out_channels=80, padding='VALID', scale=False, verbose=True, scope='conv_3')
            x = self.conv_bn_act(x, 3, 1, out_channels=192, padding='VALID', scale=False, verbose=True, scope='conv_4')
            x = self.max_pool(x, 3, 2, padding='VALID')
        d['block_{}'.format(self._curr_block)] = x

        self._curr_block += 1
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.inception_a(x, 32, name='inception_a_0')
            x = self.inception_a(x, 64, name='inception_a_1')
            x = self.inception_a(x, 64, name='inception_a_2')

        self._curr_block += 1
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.reduction_a(x, name='reduction_a')

        self._curr_block += 1
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.inception_b(x, 128, name='inception_b_0')
            x = self.inception_b(x, 160, name='inception_b_1')
            x = self.inception_b(x, 160, name='inception_b_2')
            x = self.inception_b(x, 192, name='inception_b_3')
        aux_branch = x

        self._curr_block += 1
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.reduction_b(x, name='reduction_b')

        self._curr_block += 1
        with tf.variable_scope(f'block_{self._curr_block}'):
            x = self.inception_c(x, name='inception_c_0')
            x = self.inception_c(x, name='inception_c_1')

        if self.backbone_only is False:
            self._curr_block = None
            with tf.variable_scope('block_{}'.format(self._curr_block)):
                if self.has_aux_logits:
                    aux_logits = self.inception_aux(aux_branch, name='inception_aux')
                    if self.dtype is not tf.float32:
                        aux_logits = tf.cast(aux_logits, dtype=tf.float32)
                    self.aux_logits = aux_logits
                else:
                    self.aux_logits = None
                with tf.variable_scope('logits'):
                    axis = [2, 3] if self.channel_first else [1, 2]
                    x = tf.reduce_mean(x, axis=axis)
                    d['logits' + '/avgpool'] = x
                    x = tf.nn.dropout(x, rate=self.dropout_rate_features)
                    x = self.fc_layer(x, self.num_classes, verbose=True)
                    d['logits'] = x
                    d['pred'] = tf.nn.softmax(x)
        return d

    def _loss_fn(self, labels, logits, **kwargs):
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, axis=-1)
        if self.aux_logits is not None:
            softmax_losses += tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.aux_logits, axis=-1)
        return softmax_losses

    def inception_a(self, x_in, pool_channels, name='inception_a'):
        with tf.variable_scope(name):
            with tf.variable_scope('branch_1x1'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 64, scale=False, verbose=True, scope='conv_0')
            branch_1x1 = x

            with tf.variable_scope('branch_5x5'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 48, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, 5, 1, 64, scale=False, verbose=True, scope='conv_1')
            branch_5x5 = x

            with tf.variable_scope('branch_3x3'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 64, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, 3, 1, 96, scale=False, verbose=True, scope='conv_1')
                x = self.conv_bn_act(x, 3, 1, 96, scale=False, verbose=True, scope='conv_2')
            branch_3x3 = x

            with tf.variable_scope('branch_pool'):
                x = x_in
                x = self.avg_pool(x, 3, 1)
                x = self.conv_bn_act(x, 1, 1, pool_channels, scale=False, verbose=True, scope='conv_0')
            branch_pool = x

        axis = 1 if self.channel_first else -1
        return tf.concat([branch_1x1, branch_5x5, branch_3x3, branch_pool], axis=axis)

    def reduction_a(self, x_in, name='reduction_a'):
        with tf.variable_scope(name):
            with tf.variable_scope('branch_3x3_0'):
                x = x_in
                x = self.conv_bn_act(x, 3, 2, 384, padding='VALID', scale=False, verbose=True, scope='conv_0')
            branch_3x3_0 = x

            with tf.variable_scope('branch_3x3_1'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 64, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, 3, 1, 96, scale=False, verbose=True, scope='conv_1')
                x = self.conv_bn_act(x, 3, 2, 96, padding='VALID', scale=False, verbose=True, scope='conv_2')
            branch_3x3_1 = x

            with tf.variable_scope('branch_pool'):
                x = x_in
                x = self.max_pool(x, 3, 2, padding='VALID')
            branch_pool = x

        axis = 1 if self.channel_first else -1
        return tf.concat([branch_3x3_0, branch_3x3_1, branch_pool], axis=axis)

    def inception_b(self, x_in, channels_7x7, name='inception_b'):
        c7 = channels_7x7
        with tf.variable_scope(name):
            with tf.variable_scope('branch_1x1'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 192, scale=False, verbose=True, scope='conv_0')
            branch_1x1 = x

            with tf.variable_scope('branch_7x7_0'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, c7, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, (1, 7), 1, c7, scale=False, verbose=True, scope='conv_1')
                x = self.conv_bn_act(x, (7, 1), 1, 192, scale=False, verbose=True, scope='conv_2')
            branch_7x7_0 = x

            with tf.variable_scope('branch_7x7_1'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, c7, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, (7, 1), 1, c7, scale=False, verbose=True, scope='conv_1')
                x = self.conv_bn_act(x, (1, 7), 1, c7, scale=False, verbose=True, scope='conv_2')
                x = self.conv_bn_act(x, (7, 1), 1, c7, scale=False, verbose=True, scope='conv_3')
                x = self.conv_bn_act(x, (1, 7), 1, 192, scale=False, verbose=True, scope='conv_4')
            branch_7x7_1 = x

            with tf.variable_scope('branch_pool'):
                x = x_in
                x = self.avg_pool(x, 3, 1)
                x = self.conv_bn_act(x, 1, 1, 192, scale=False, verbose=True, scope='conv_0')
            branch_pool = x

        axis = 1 if self.channel_first else -1
        return tf.concat([branch_1x1, branch_7x7_0, branch_7x7_1, branch_pool], axis=axis)

    def reduction_b(self, x_in, name='reduction_b'):
        with tf.variable_scope(name):
            with tf.variable_scope('branch_3x3'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 192, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, 3, 2, 320, padding='VALID', scale=False, verbose=True, scope='conv_1')
            branch_3x3 = x

            with tf.variable_scope('branch_7x7'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 192, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, (1, 7), 1, 192, scale=False, verbose=True, scope='conv_1')
                x = self.conv_bn_act(x, (7, 1), 1, 192, scale=False, verbose=True, scope='conv_2')
                x = self.conv_bn_act(x, 3, 2, 192, padding='VALID', scale=False, verbose=True, scope='conv_3')
            branch_7x7 = x

            with tf.variable_scope('branch_pool'):
                x = x_in
                x = self.max_pool(x, 3, 2, padding='VALID')
            branch_pool = x

        axis = 1 if self.channel_first else -1
        return tf.concat([branch_3x3, branch_7x7, branch_pool], axis=axis)

    def inception_c(self, x_in, name='inception_c'):
        with tf.variable_scope(name):
            with tf.variable_scope('branch_1x1'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 320, scale=False, verbose=True, scope='conv_0')
            branch_1x1 = x

            with tf.variable_scope('branch_3x3_0'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 384, scale=False, verbose=True, scope='conv_0')
                branch = x
                x = self.conv_bn_act(branch, (1, 3), 1, 384, scale=False, verbose=True, scope='conv_1a')
                branch_3x3_0a = x
                x = self.conv_bn_act(branch, (3, 1), 1, 384, scale=False, verbose=True, scope='conv_1b')
                branch_3x3_0b = x

            with tf.variable_scope('branch_3x3_1'):
                x = x_in
                x = self.conv_bn_act(x, 1, 1, 448, scale=False, verbose=True, scope='conv_0')
                x = self.conv_bn_act(x, 3, 1, 384, scale=False, verbose=True, scope='conv_1')
                branch = x
                x = self.conv_bn_act(branch, (1, 3), 1, 384, scale=False, verbose=True, scope='conv_2a')
                branch_3x3_1a = x
                x = self.conv_bn_act(branch, (3, 1), 1, 384, scale=False, verbose=True, scope='conv_2b')
                branch_3x3_1b = x

            with tf.variable_scope('branch_pool'):
                x = x_in
                x = self.avg_pool(x, 3, 1)
                x = self.conv_bn_act(x, 1, 1, 192, scale=False, verbose=True, scope='conv_0')
            branch_pool = x

        axis = 1 if self.channel_first else -1
        return tf.concat([branch_1x1, branch_3x3_0a, branch_3x3_0b, branch_3x3_1a, branch_3x3_1b, branch_pool],
                         axis=axis)

    def inception_aux(self, x, name='inception_aux'):
        with tf.variable_scope(name):
            x = self.avg_pool(x, 5, 3, padding='VALID')
            x = self.conv_bn_act(x, 1, 1, 128, scale=False, verbose=True, scope='conv_0')
            x = self.conv_bn_act(x, 5, 1, 768, scale=False, verbose=True, scope='conv_1')
            with tf.variable_scope('logits'):
                axis = [2, 3] if self.channel_first else [1, 2]
                x = tf.reduce_mean(x, axis=axis)
                x = self.fc_layer(x, self.num_classes, verbose=True)
            return x
