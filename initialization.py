import os
import argparse
import ast


def init_params(d, parser=None):
    """
    :param d: dict, (hyper)parameters
    :param parser: argparse.ArgumentParser object
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    _, unknown_args = parser.parse_known_args()

    is_arg = False
    arg_name = ''
    for arg in unknown_args:
        if is_arg and not arg.startswith('--'):
            try:
                arg = ast.literal_eval(arg)
                is_string = False
            except ValueError:
                is_string = True
            d[arg_name] = arg
            is_arg = False
            if is_string:
                print(f'"{arg_name}":', f'"{arg}"')
            else:
                print(f'"{arg_name}":', arg)
        else:
            if arg.startswith('--'):
                arg_name = arg.lstrip('-')
                is_arg = True
            elif arg.startswith('-'):
                raise ValueError(f'Argument names must start with "--" ({arg}).')
    print()

    gpu_offset = d.get('gpu_offset', 0)
    num_gpus = d.get('num_gpus', 1)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(num) for num
                                                  in range(gpu_offset, num_gpus + gpu_offset))
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(num_gpus)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    # os.environ["OMP_NUM_THREADS"] = str(d['num_parallel_calls'])
    # os.environ["KMP_BLOCKTIME"] = '0'
    # os.environ["KMP_SETTINGS"] = '1'
    # os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
