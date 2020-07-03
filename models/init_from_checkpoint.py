"""
Initialize your networks using public checkpoints.
Define mapping rules to load variables in the checkpoints
ResNet v1: https://github.com/tensorflow/models/tree/master/research/slim
ResNet v2: https://github.com/tensorflow/models/tree/master/research/slim
VGGNet: https://github.com/tensorflow/models/tree/master/research/slim
"""

import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_tensorflow


def resnet_v1_50_101(ckpt_dir, model_scope=None, load_moving_average=True, verbose=True):
    if ckpt_dir.split('_')[-1] == '50.ckpt':
        prefix = 'resnet_v1_50/'
    else:
        prefix = 'resnet_v1_101/'
    start_idx = [0, 1, 1]   # Starting indices for any blocks, units, and convolutions

    exception_blocks = ['0', 'None']
    exception_convs = ['skip']

    index_offsets = {}

    block_prefix = 'block'
    key_match_dict = {block_prefix: 'block',
                      'res': 'unit_',
                      'conv': 'bottleneck_v1/conv',
                      'bn': 'BatchNorm',
                      'weights': 'weights',
                      'biases': 'biases',
                      'mu': 'moving_mean',
                      'sigma': 'moving_variance',
                      'beta': 'beta',
                      'gamma': 'gamma',
                      'ExponentialMovingAverage': 'ExponentialMovingAverage'}

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_names = [var for var in var_to_shape_map.keys()]

    global_variables = tf.global_variables()
    variables = []
    for var in global_variables:
        if model_scope is None:
            variables.append(var)
        else:
            if var.name.startswith(model_scope):
                variables.append(var)
    variables_not_loaded = [var.name for var in variables]
    assign_dict = dict()
    for var in variables:
        keys = var.name.rstrip(':0').split('/')
        if model_scope is not None:
            keys = keys[1:]
        if keys[-1] in key_match_dict:
            var_name_splitted = []
            block_name = keys[0].split('_')[-1]
            if block_name in exception_blocks:  # FIXME
                if block_name == '0':
                    part_name = keys[2]
                    if part_name == 'bn':
                        conv_num = keys[1].split('_')[-1]
                        var_name_splitted.append('conv' + str(int(conv_num) + start_idx[2]))
                        var_name_splitted.append('BatchNorm')
                    else:
                        conv_num = keys[1].split('_')[-1]
                        var_name_splitted.append('conv' + str(int(conv_num) + start_idx[2]))
                elif block_name == 'None':
                    var_name_splitted.append('logits')
                else:
                    raise ValueError('block_{} is not considered as an exception'.format(block_name))
                if keys[-1] == 'ExponentialMovingAverage':  # To load exponential moving averages
                    var_name_splitted.append(key_match_dict[keys[-2]])
                var_name_splitted.append(key_match_dict[keys[-1]])
            else:
                unit_name = '_'.join(keys[1].split('_')[:-1])
                unit_num = keys[1].split('_')[-1]
                conv_num = keys[2].split('_')[-1]
                if conv_num in exception_convs:  # FIXME
                    if unit_name in key_match_dict:
                        if conv_num == 'skip':
                            var_name_splitted.append(key_match_dict[block_prefix] + str(int(block_name) + start_idx[0]))
                            var_name_splitted.append(key_match_dict[unit_name] + str(int(unit_num) + start_idx[1]))
                            var_name_splitted.append('bottleneck_v1/shortcut')
                            if keys[3] == 'bn':
                                var_name_splitted.append('BatchNorm')
                        else:
                            raise (ValueError, 'conv_{} is not considered as an exception'.format(conv_num))
                        if keys[-1] == 'ExponentialMovingAverage':  # To load exponential moving averages
                            var_name_splitted.append(key_match_dict[keys[-2]])
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
                            var_name_splitted.append(part_name + str(key_num + start_idx[i] + offset))

            var_name = prefix + '/'.join(var_name_splitted)
            if load_moving_average:
                if var_name + '/ExponentialMovingAverage' in var_to_shape_map:
                    if var.get_shape() == var_to_shape_map[var_name + '/ExponentialMovingAverage']:
                        assign_dict[var_name] = var
                        if var_name + '/ExponentialMovingAverage' in var_names:
                            var_names.remove(var_name + '/ExponentialMovingAverage')
                        variables_not_loaded.remove(var.name)
                        if verbose:
                            print('Init. {} <===== {}'.format(var.name, var_name + '/ExponentialMovingAverage'))
                    elif verbose:
                        print('Init. {} <==/== {} (variable shapes do not match)'
                              .format(var.name, var_name + '/ExponentialMovingAverage'))
                elif var_name in var_to_shape_map:
                    if var.get_shape() == var_to_shape_map[var_name]:
                        assign_dict[var_name] = var
                        if var_name in var_names:
                            var_names.remove(var_name)
                        variables_not_loaded.remove(var.name)
                        if verbose:
                            print('Init. {} <===== {}'.format(var.name, var_name))
                    elif verbose:
                        print('Init. {} <==/== {} (variable shapes do not match)'.format(var.name, var_name))
            else:
                if var_name in var_to_shape_map:
                    if var.get_shape() == var_to_shape_map[var_name]:
                        assign_dict[var_name] = var
                        var_names.remove(var_name)
                        variables_not_loaded.remove(var.name)
                        if verbose:
                            print('Init. {} <===== {}'.format(var.name, var_name))
                    elif verbose:
                        print('Init. {} <==/== {} (variable shapes do not match)'.format(var.name, var_name))

    tf.train.init_from_checkpoint(ckpt_dir, assign_dict)

    print('')
    print('Variables have been initialized using the following checkpoint:')
    print(ckpt_dir)
    print('The following variables in the checkpoint were not used:')
    print(var_names)
    print('The following variables do not exist in the checkpoint, so they were initialized randomly:')
    print(variables_not_loaded)
    print('')


def resnet_v2_50_101(ckpt_dir, model_scope=None, load_moving_average=True, verbose=True):
    if ckpt_dir.split('_')[-1] == '50.ckpt':
        prefix = 'resnet_v2_50/'
    else:
        prefix = 'resnet_v2_101/'
    start_idx = [0, 1, 1]   # Starting indices for any blocks, units, and convolutions

    exception_blocks = ['0', 'None']
    exception_convs = ['0', 'skip']

    index_offsets = {'bn': -1}

    block_prefix = 'block'
    key_match_dict = {block_prefix: 'block',
                      'res': 'unit_',
                      'conv': 'bottleneck_v2/conv',
                      'bn': 'BatchNorm',
                      'weights': 'weights',
                      'biases': 'biases',
                      'mu': 'moving_mean',
                      'sigma': 'moving_variance',
                      'beta': 'beta',
                      'gamma': 'gamma',
                      'ExponentialMovingAverage': 'ExponentialMovingAverage'}

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_names = [var for var in var_to_shape_map.keys()]

    global_variables = tf.global_variables()
    variables = []
    for var in global_variables:
        if model_scope is None:
            variables.append(var)
        else:
            if var.name.startswith(model_scope):
                variables.append(var)
    variables_not_loaded = [var.name for var in variables]
    assign_dict = dict()
    for var in variables:
        keys = var.name.rstrip(':0').split('/')
        if model_scope is not None:
            keys = keys[1:]
        if keys[-1] in key_match_dict:
            var_name_splitted = []
            block_name = keys[0].split('_')[-1]
            if block_name in exception_blocks:  # FIXME
                if block_name == '0':
                    conv_num = keys[1].split('_')[-1]
                    var_name_splitted.append('conv' + str(int(conv_num) + start_idx[2]))
                elif block_name == 'None':
                    if keys[2] == 'bn':
                        var_name_splitted.append('postnorm')
                    else:
                        var_name_splitted.append('logits')
                else:
                    raise ValueError('block_{} is not considered as an exception'.format(block_name))
                if keys[-1] == 'ExponentialMovingAverage':  # To load exponential moving averages
                    var_name_splitted.append(key_match_dict[keys[-2]])
                var_name_splitted.append(key_match_dict[keys[-1]])
            else:
                unit_name = '_'.join(keys[1].split('_')[:-1])
                unit_num = keys[1].split('_')[-1]
                conv_name = '_'.join(keys[2].split('_')[:-1])
                conv_num = keys[2].split('_')[-1]
                if conv_num in exception_convs:  # FIXME
                    if unit_name in key_match_dict:
                        if conv_num == '0':
                            var_name_splitted.append(key_match_dict[block_prefix] + str(int(block_name) + start_idx[0]))
                            var_name_splitted.append(key_match_dict[unit_name] + str(int(unit_num) + start_idx[1]))
                            if 'bn' in keys:
                                var_name_splitted.append('bottleneck_v2/preact')
                            else:
                                var_name_splitted.append(key_match_dict[conv_name] + str(int(conv_num) + start_idx[2]))
                        elif conv_num == 'skip':
                            var_name_splitted.append(key_match_dict[block_prefix] + str(int(block_name) + start_idx[0]))
                            var_name_splitted.append(key_match_dict[unit_name] + str(int(unit_num) + start_idx[1]))
                            var_name_splitted.append('bottleneck_v2/shortcut')
                        else:
                            raise (ValueError, 'conv_{} is not considered as an exception'.format(conv_num))
                        if keys[-1] == 'ExponentialMovingAverage':  # To load exponential moving averages
                            var_name_splitted.append(key_match_dict[keys[-2]])
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
                            var_name_splitted.append(part_name + str(key_num + start_idx[i] + offset))

            var_name = prefix + '/'.join(var_name_splitted)
            if load_moving_average:
                if var_name + '/ExponentialMovingAverage' in var_to_shape_map:
                    if var.get_shape() == var_to_shape_map[var_name + '/ExponentialMovingAverage']:
                        assign_dict[var_name] = var
                        if var_name + '/ExponentialMovingAverage' in var_names:
                            var_names.remove(var_name + '/ExponentialMovingAverage')
                        variables_not_loaded.remove(var.name)
                        if verbose:
                            print('Init. {} <===== {}'.format(var.name, var_name + '/ExponentialMovingAverage'))
                    elif verbose:
                        print('Init. {} <==/== {} (variable shapes do not match)'
                              .format(var.name, var_name + '/ExponentialMovingAverage'))
                elif var_name in var_to_shape_map:
                    if var.get_shape() == var_to_shape_map[var_name]:
                        assign_dict[var_name] = var
                        if var_name in var_names:
                            var_names.remove(var_name)
                        variables_not_loaded.remove(var.name)
                        if verbose:
                            print('Init. {} <===== {}'.format(var.name, var_name))
                    elif verbose:
                        print('Init. {} <==/== {} (variable shapes do not match)'.format(var.name, var_name))
            else:
                if var_name in var_to_shape_map:
                    if var.get_shape() == var_to_shape_map[var_name]:
                        assign_dict[var_name] = var
                        var_names.remove(var_name)
                        variables_not_loaded.remove(var.name)
                        if verbose:
                            print('Init. {} <===== {}'.format(var.name, var_name))
                    elif verbose:
                        print('Init. {} <==/== {} (variable shapes do not match)'.format(var.name, var_name))

    tf.train.init_from_checkpoint(ckpt_dir, assign_dict)

    print('')
    print('Variables have been initialized using the following checkpoint:')
    print(ckpt_dir)
    print('The following variables in the checkpoint were not used:')
    print(var_names)
    print('The following variables do not exist in the checkpoint, so they were initialized randomly:')
    print(variables_not_loaded)
    print('')


def vggnet(ckpt_dir, model_scope=None, load_moving_average=False, verbose=True):
    if load_moving_average:
        print('No moving average exists for VGGNet.')

    if ckpt_dir.split('_')[-1] == '16.ckpt':
        prefix = 'vgg_16/'
    else:
        prefix = 'vgg_19/'
    start_idx = [1, 1]   # Starting indices for any blocks and convolutions
    exception_blocks = ['None']

    block_prefix = 'block'
    key_match_dict = {block_prefix: 'conv',
                      'conv': 'conv',
                      'fc_0': 'fc6',
                      'fc_1': 'fc7',
                      'fc_2': 'fc8',
                      'weights': 'weights',
                      'biases': 'biases'}

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_names = [var for var in var_to_shape_map.keys()]

    global_variables = tf.global_variables()
    variables = []
    for var in global_variables:
        if model_scope is None:
            variables.append(var)
        else:
            if var.name.startswith(model_scope):
                variables.append(var)
    variables_not_loaded = [var.name for var in variables]
    assign_dict = dict()
    for var in variables:
        keys = var.name.rstrip(':0').rstrip('/ExponentialMovingAverage').split('/')
        if model_scope is not None:
            keys = keys[1:]
        if keys[-1] in key_match_dict:
            var_name_splitted = []
            block_name = keys[0].split('_')[-1]
            if block_name in exception_blocks:  # FIXME
                if block_name == 'None':
                    var_name_splitted.append(key_match_dict[keys[-2]])
                else:
                    raise ValueError('block_{} is not considered as an exception'.format(block_name))
            else:
                block_num = str(int(block_name) + start_idx[0])
                var_name_splitted.append(key_match_dict[block_prefix] + block_num)
                conv_name = keys[-2].split('_')[0]
                conv_num = str(int(keys[-2].split('_')[-1]) + start_idx[1])
                var_name_splitted.append(key_match_dict[conv_name] + block_num + '_' + conv_num)
            var_name_splitted.append(key_match_dict[keys[-1]])

            var_name = prefix + '/'.join(var_name_splitted)
            if var_name in var_to_shape_map:
                if var.get_shape() == var_to_shape_map[var_name]:
                    assign_dict[var_name] = var
                    if var_name in var_names:
                        var_names.remove(var_name)
                    variables_not_loaded.remove(var.name)
                    if verbose:
                        print('Init. {} <===== {}'.format(var.name, var_name))
                elif verbose:
                    print('Init. {} <==/== {} (variable shapes do not match)'.format(var.name, var_name))

    tf.train.init_from_checkpoint(ckpt_dir, assign_dict)

    print('')
    print('Variables have been initialized using the following checkpoint:')
    print(ckpt_dir)
    print('The following variables in the checkpoint were not used:')
    print(var_names)
    print('The following variables do not exist in the checkpoint, so they were initialized randomly:')
    print(variables_not_loaded)
    print('')
