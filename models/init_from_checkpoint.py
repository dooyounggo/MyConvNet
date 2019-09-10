import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def resnet_v2_50_101(ckpt_dir, load_moving_average=True):     # https://github.com/tensorflow/models/tree/master/research/slim
    if ckpt_dir.split('_')[-1] == '50.ckpt':
        prefix = 'resnet_v2_50/'
    else:
        prefix = 'resnet_v2_101/'
    start_indices = [0, 1, 1]   # Starting indices for any blocks, units, and convolutions

    exception_blocks = ['0', 'None']
    exception_convs = ['0', 'skip']

    index_offsets = {'bn': -1}

    key_match_dict = {'block': 'block',
                      'res': 'unit_',
                      'conv': 'bottleneck_v2/conv',
                      'bn': 'BatchNorm',
                      'weights:0': 'weights',
                      'biases:0': 'biases',
                      'mu:0': 'moving_mean',
                      'sigma:0': 'moving_variance',
                      'beta:0': 'beta',
                      'gamma:0': 'gamma'}

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()

    variables = tf.global_variables()
    assign_dict = dict()
    for var in variables:
        keys = var.name.split('/')
        if keys[-1] in key_match_dict:
            var_name_splitted = []
            block_name = '_'.join(keys[0].split('_')[:-1])
            block_num = keys[0].split('_')[-1]
            if block_num in exception_blocks:       # FIXME
                if block_num == '0':
                    conv_num = keys[1].split('_')[-1]
                    var_name_splitted.append('conv' + str(int(conv_num) + start_indices[2]))
                elif block_num == 'None':
                    if keys[2] == 'bn':
                        var_name_splitted.append('postnorm')
                    else:
                        var_name_splitted.append('logits_are_not_loaded')   # logits
                else:
                    raise(ValueError, 'block_{} is not considered as an exception'.format(block_num))
                var_name_splitted.append(key_match_dict[keys[-1]])
            else:
                unit_name = '_'.join(keys[1].split('_')[:-1])
                unit_num = keys[1].split('_')[-1]
                conv_name = '_'.join(keys[2].split('_')[:-1])
                conv_num = keys[2].split('_')[-1]
                if conv_num in exception_convs:     # FIXME
                    if unit_name in key_match_dict:
                        if conv_num == '0':
                            var_name_splitted.append(key_match_dict[block_name] + str(int(block_num) + start_indices[0]))
                            var_name_splitted.append(key_match_dict[unit_name] + str(int(unit_num) + start_indices[1]))
                            if keys[-1] == 'weights:0' or keys[-1] == 'biases:0':
                                var_name_splitted.append(key_match_dict[conv_name] + str(int(conv_num) + start_indices[2]))
                            else:
                                var_name_splitted.append('bottleneck_v2/preact')
                        elif conv_num == 'skip':
                            var_name_splitted.append(key_match_dict[block_name] + str(int(block_num) + start_indices[0]))
                            var_name_splitted.append(key_match_dict[unit_name] + str(int(unit_num) + start_indices[1]))
                            var_name_splitted.append('bottleneck_v2/shortcut')
                        else:
                            raise (ValueError, 'conv_{} is not considered as an exception'.format(conv_num))
                        var_name_splitted.append(key_match_dict[keys[-1]])
                else:
                    for i, key in enumerate(keys):
                        key_name = '_'.join(key.split('_')[:-1])
                        try:
                            key_num = int(key.split('_')[-1])
                        except ValueError:
                            key_num = None
                            key_name = key

                        if key_name in key_match_dict:
                            part_name = key_match_dict[key_name]
                        else:
                            part_name = 'InvalidKey:{}'.format(key_name)

                        if i < len(keys) - 1:
                            next_key_name = '_'.join(keys[i + 1].split('_')[:-1])
                            if next_key_name == '':
                                next_key_name = keys[i + 1]
                        else:
                            next_key_name = None

                        offset = index_offsets.get(next_key_name, 0)
                        if key_num is None:
                            var_name_splitted.append(part_name)
                        else:
                            var_name_splitted.append(part_name + str(key_num + start_indices[i] + offset))

            var_name = prefix + '/'.join(var_name_splitted)
            if load_moving_average:
                if var_name + '/ExponentialMovingAverage' in var_to_shape_map:
                    print('Init. {} <===== {}'.format(var.name, var_name + '/ExponentialMovingAverage'))
                    assign_dict[var_name] = var
                elif var_name in var_to_shape_map:
                    print('Init. {} <===== {}'.format(var.name, var_name))
                    assign_dict[var_name] = var
            else:
                if var_name in var_to_shape_map:
                    print('Init. {} <===== {}'.format(var.name, var_name))
                    assign_dict[var_name] = var

    tf.train.init_from_checkpoint(ckpt_dir, assign_dict)
