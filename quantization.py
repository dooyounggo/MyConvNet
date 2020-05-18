"""
Model quantization.
Note that only the float32 data type and the "NHWC" format are supported.
"""

import os
import sys
import pathlib
import shutil
import time
import ast
import pydot
import numpy as np
import tensorflow.compat.v1 as tf
import multiprocessing as mp
from subsets.subset_functions import resize_with_crop_or_pad


def quantize(model, images, ckpt_dir, save_dir, overwrite=False, saved_model=True, **kwargs):
    """
    :param model: ConvNet, a model to be quantized.
    :param images: np.ndarray, representative images used for quantization.
    :param ckpt_dir: string, a path to saved checkpoint.
    :param save_dir: string, a path to save models.
    :param overwrite: bool, whether to overwrite tflite files already exist.
    :param saved_model: bool, Whether to create saved_model from ckpt.
    :param kwargs: hyperparameters.
    :return: paths to a tflite model file and a quantized tflite model file.
    """
    model.close()
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    sess = tf.Session(graph=graph, config=config)
    model._curr_device = 0
    with tf.device(model.param_device):
        # model.is_train = tf.constant(False, dtype=tf.bool, name='is_train')
        model.is_train = False
        model.dropout_rate_weights = tf.constant(0.0, dtype=model.dtype, name='dropout_rate_weights')
        model.dropout_rate_features = tf.constant(0.0, dtype=model.dtype, name='dropout_rate_features')
        model.ema = tf.train.ExponentialMovingAverage(decay=model.moving_average_decay)
    with tf.device('/{}:0'.format(model.compute_device)):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=([None] + list(model.input_size)), name='input')
        model.X = input_tensor
        d = model._build_model(**kwargs)
        output_tensor = d['pred']

    output_tensors = [output_tensor]
    operations = graph.get_operations()
    act_names = ['relu', 'swish', 'tanh', 'sigmoid']
    for op in operations:
        op_tensors = op.values()
        if len(op_tensors) > 0:
            op_tensor = op_tensors[0]
            for act in act_names:
                if act in op_tensor.name.lower():
                    output_tensors.append(op_tensor)
                    break
    for n in range(model.num_blocks):
        if f'block_{n}' in d:
            output_tensors.append(d[f'block_{n}'])

    if kwargs.get('zero_center', True):
        image_mean = kwargs.get('image_mean', 0.5)
    else:
        image_mean = 0.0
    scale_factor = kwargs.get('scale_factor', 2.0)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess, ckpt_dir)
    converter = tf.lite.TFLiteConverter.from_session(sess=sess,
                                                     input_tensors=[input_tensor],
                                                     output_tensors=output_tensors)

    tflite_models_dir = pathlib.Path(os.path.join(save_dir, 'tflite'))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/'model.tflite'
    tflite_model_quant_file = tflite_models_dir/'model_quantized.tflite'

    if overwrite or not tflite_model_file.exists():
        print('Converting the model ...')
        tflite_graphviz_dir = tflite_models_dir/'graphviz'
        tflite_graphviz_dir.mkdir(exist_ok=True, parents=True)
        converter.dump_graphviz_dir = str(tflite_graphviz_dir)
        tflite_model = converter.convert()
        tflite_model_file.write_bytes(tflite_model)
        if os.name != 'nt':  # Conversion from dot to svg is not supported on Windows due to UnicodeDecodeError.
            dotfile_names = ['toco_AT_IMPORT', 'toco_AFTER_TRANSFORMATIONS', 'toco_AFTER_ALLOCATION']
            for dotname in dotfile_names:
                (dotgraph,) = pydot.graph_from_dot_file(os.path.join(str(tflite_graphviz_dir), dotname + '.dot'))
                dotgraph.write_svg(os.path.join(str(tflite_graphviz_dir), dotname + '.svg'))
        print('Done. \n')
    else:
        print('The tflite model already exists.')

    if overwrite or not tflite_model_quant_file.exists():
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

        def repr_data_gen():
            for img in images:
                yield [(img[np.newaxis, ...] - image_mean)*scale_factor]

        converter.representative_dataset = repr_data_gen

        print('Converting the quantized model ...')
        converter.dump_graphviz_dir = None  # No graphviz for the quantized model since the results are identical.
        tflite_model_quant = converter.convert()
        tflite_model_quant_file.write_bytes(tflite_model_quant)
        print('Done. \n')
    else:
        print('The quantized tflite model already exists.')

    if saved_model:
        saved_model_dir = os.path.join(save_dir, 'saved_model')
        if os.path.exists(saved_model_dir):
            print('Remove existing saved_model files ...', end=' ')
            shutil.rmtree(saved_model_dir)
        print('Converting the ckpt to saved_model ...', end=' ')
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(sess, tags=tf.saved_model.SERVING, strip_default_attrs=True)
        builder.save()
        print('Done. \n')

    return tflite_model_file, tflite_model_quant_file


def evaluate_quantized_model(model_file, model_quant_file, test_set, evaluator, num_processes=4, **kwargs):
    mp.set_start_method('spawn')
    dataset = tf.data.Dataset.from_tensor_slices((test_set.image_dirs, test_set.label_dirs))
    if not test_set.from_memory:
        dataset = dataset.map(lambda image_dir, label_dir: tuple(tf.py_func(test_set._load_function,
                                                                            (image_dir, label_dir),
                                                                            (tf.float32, tf.float32))),
                              num_parallel_calls=1)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    session = tf.Session()

    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()
    interpreter_quant = tf.lite.Interpreter(model_path=str(model_quant_file))
    interpreter_quant.allocate_tensors()

    input_details_quant = interpreter_quant.get_input_details()[0]
    output_details_quant = interpreter_quant.get_output_details()[0]

    image_mean = kwargs.get('image_mean', 0.5)
    scale_factor = kwargs.get('scale_factor', 2.0)
    argmax_output = kwargs.get('argmax_output', False)
    num_prints = kwargs.get('num_prints', 100)

    params = dict()
    params['model_file'] = model_file
    params['model_quant_file'] = model_quant_file
    params['num_images'] = test_set.num_examples
    params['image_mean'] = image_mean
    params['scale_factor'] = scale_factor
    params['num_processes'] = num_processes
    params['argmax_output'] = argmax_output
    params['num_prints'] = num_prints

    label_shape = [test_set.num_examples] + list(output_details_quant['shape'][1:-1])
    if argmax_output:
        output_shape = label_shape
    else:
        output_shape = [test_set.num_examples] + list(output_details_quant['shape'][1:])
    idx = mp.Value('l', lock=False)
    image_arr = mp.Array('f', int(np.prod(input_details_quant['shape'][1:])), lock=False)
    gt_label_arr = mp.Array('f', int(np.prod(label_shape)), lock=False)
    results_arr = mp.Array('f', int(np.prod(output_shape)), lock=False)
    results_quant_arr = mp.Array('f', int(np.prod(output_shape)), lock=False)
    w_lock = mp.Lock()
    r_lock = mp.Lock()

    r_lock.acquire()  # Lock image read at the beginning
    procs = []
    for n in range(num_processes):
        proc = mp.Process(target=tflite_process, name='invoke_process_{}'.format(n),
                          args=(idx, image_arr, results_arr, results_quant_arr, w_lock, r_lock), kwargs=params)
        proc.start()
        procs.append(proc)

    load_function(idx, image_arr, gt_label_arr, w_lock, r_lock, session, iterator, interpreter, **params)

    for proc in procs:
        proc.join()

    gt_label = np.empty(label_shape, dtype=np.float32)
    results = np.empty(output_shape, dtype=np.float32)
    results_quant = np.empty(output_shape, dtype=np.float32)
    gt_len = np.prod(label_shape[1:]).astype(int)
    r_len = np.prod(output_shape[1:]).astype(int)
    for i in range(test_set.num_examples):
        gt_label[i] = np.array(gt_label_arr[i*gt_len:(i + 1)*gt_len]).reshape(label_shape[1:])
        results[i] = np.array(results_arr[i*r_len:(i + 1)*r_len]).reshape(output_shape[1:])
        results_quant[i] = np.array(results_quant_arr[i*r_len:(i + 1)*r_len]).reshape(output_shape[1:])

    gt_label = gt_label[..., np.newaxis]
    if argmax_output:
        results = results[..., np.newaxis]
        results_quant = results_quant[..., np.newaxis]

    accuracy = evaluator.score(gt_label, results)
    accuracy_quant = evaluator.score(gt_label, results_quant)
    if output_shape[-1] == 1:
        output_quant_details = output_details_quant['quantization']
        if output_quant_details[0] > 0.0:
            results_quant_float = results_quant.astype(np.float32)/output_quant_details[0] + output_quant_details[1]
        else:
            results_quant_float = results_quant.astype(np.float32)
        is_different = np.logical_not(np.isclose(results, results_quant_float, rtol=1.e-4, atol=0))
    else:
        is_different = np.not_equal(np.argmax(results, axis=-1), np.argmax(results_quant, axis=-1))

    print('\nAccuracy Before Quantization: {:.4f}'.format(accuracy))
    print('Accuracy After Quantization:  {:.4f}'.format(accuracy_quant))
    print('Number of Different Results: {}/{}'.format(np.sum(is_different), np.prod(is_different.shape)))


def load_function(idx, image, gt_label, w_lock, r_lock, session, iterator, interpreter, **kwargs):
    num_images = kwargs['num_images']
    image_mean = kwargs['image_mean']
    scale_factor = kwargs['scale_factor']
    num_prints = kwargs['num_prints']

    image_shape = interpreter.get_input_details()[0]['shape'][1:]
    label_len = np.prod(interpreter.get_output_details()[0]['shape'][1:-1]).astype(int)
    image_np = np.frombuffer(image, dtype=np.float32)
    label_np = np.frombuffer(gt_label, dtype=np.float32)

    session.run(iterator.initializer)
    image_tensor, label_tensor = iterator.get_next()
    while True:
        w_lock.acquire()
        i = idx.value
        idx.value = i + 1
        if i >= num_images:
            w_lock.release()
            r_lock.release()
            break
        else:
            input_image, input_label = session.run([image_tensor, label_tensor])
            input_image = resize_with_crop_or_pad(input_image[0],
                                                  out_size=image_shape)
            input_image = (input_image - image_mean)*scale_factor
            image_np[:] = np.reshape(input_image, np.prod(image_shape).astype(int))
            label_np[i*label_len:(i + 1)*label_len] = np.reshape(input_label, label_len)
            r_lock.release()

            if i < num_prints:
                print('{}. GT:'.format(i))
                print(input_label[0])


def tflite_process(idx, image, results, results_quant, w_lock, r_lock, **kwargs):
    model_file = kwargs['model_file']
    model_quant_file = kwargs['model_quant_file']
    num_images = kwargs['num_images']
    num_processes = kwargs['num_processes']
    argmax_output = kwargs['argmax_output']
    num_prints = kwargs['num_prints']

    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()
    interpreter_quant = tf.lite.Interpreter(model_path=str(model_quant_file))
    interpreter_quant.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_details_quant = interpreter_quant.get_input_details()[0]
    output_details_quant = interpreter_quant.get_output_details()[0]
    input_index_quant = input_details_quant['index']
    output_index_quant = output_details_quant['index']

    results_np = np.frombuffer(results, dtype=np.float32)
    results_quant_np = np.frombuffer(results_quant, dtype=np.float32)
    image_shape = interpreter_quant.get_input_details()[0]['shape'][1:]
    if argmax_output:
        out_len = np.prod(interpreter_quant.get_output_details()[0]['shape'][1:-1]).astype(int)
    else:
        out_len = np.prod(interpreter_quant.get_output_details()[0]['shape'][1:]).astype(int)
    i_local = 0
    total_time = 0
    print('Start {} (PID: {}).'.format(mp.current_process().name, os.getpid()))
    sys.stdout.flush()
    while True:
        r_lock.acquire(timeout=60)  # If you kill the main process, child processes will terminate in about 60 seconds.
        i = idx.value - 1
        if i >= num_images:
            r_lock.release()
            break
        else:
            if (i % num_prints) == 0:
                print('Evaluating models ... {:5d}/{}'.format(i, num_images))
                sys.stdout.flush()
            t_start = time.time()
            input_image = np.array(image, dtype=np.float32).reshape(image_shape)[np.newaxis, ...]
            w_lock.release()

            interpreter.set_tensor(input_index, input_image)
            interpreter.invoke()

            input_quant_details = input_details_quant['quantization']
            if input_quant_details[0] > 0.0:
                input_image_quant = input_image/input_quant_details[0] + input_quant_details[1]
            else:
                input_image_quant = input_image
            input_image_quant = input_image_quant.astype(input_details_quant['dtype'])

            interpreter_quant.set_tensor(input_index_quant, input_image_quant)
            interpreter_quant.invoke()

            output = interpreter.get_tensor(output_index)[0]
            output_quant = interpreter_quant.get_tensor(output_index_quant)[0]
            if argmax_output:
                output = np.argmax(output, axis=-1)
                output_quant = np.argmax(output_quant, axis=-1)
            results_np[i*out_len:(i + 1)*out_len] = np.reshape(output, out_len)
            results_quant_np[i*out_len:(i + 1)*out_len] = np.reshape(output_quant, out_len)

            i_local += 1
            if i < num_prints:
                total_time += time.time() - t_start
                print('Estimated test time: {} min.'.format(int(total_time/i_local*num_images/60/num_processes)))
                print('{}. Before:'.format(i))
                if output.shape[-1] == 1:
                    print(output)
                else:
                    print(np.argmax(output, axis=-1))
                print('{}. After:'.format(i))
                if output_quant.shape[-1] == 1:
                    print(output_quant)
                else:
                    print(np.argmax(output_quant, axis=-1))
                print()
                sys.stdout.flush()


def write_tensors(model_file, sample_image, tensor_list=None, with_txt=True):
    model_file = str(model_file)
    model_dir = model_file.replace('.tflite', '')
    os.makedirs(model_dir, exist_ok=True)
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    tensor_details = interpreter.get_tensor_details()

    input_quant_details = input_details['quantization']
    if input_quant_details[0] > 0.0:
        input_image_quant = sample_image/input_quant_details[0] + input_quant_details[1]
    else:
        input_image_quant = sample_image
    input_image = input_image_quant.astype(input_details['dtype'])[np.newaxis, ...]

    interpreter.set_tensor(input_details['index'], input_image)
    interpreter.invoke()

    int_types = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64']
    float_types = ['float16', 'float32', 'float64']
    print('Writing tensors for {} ...'.format(os.path.split(model_file)[1]), end=' ')
    if tensor_list is None:
        for td in tensor_details:
            name = td['name']
            main_name = os.path.join(model_dir, name)
            os.makedirs(os.path.split(main_name)[0], exist_ok=True)
            quantization = td['quantization']
            index = td['index']

            array = interpreter.get_tensor(index)
            shape = array.shape
            dtype = array.dtype
            if len(shape) == 4:
                array = np.transpose(array, [1, 2, 3, 0])
            array_bin = array.tobytes()

            with open(main_name + '.bin', mode='wb') as f:
                f.write(array_bin)
            with open(main_name + '.info', mode='w') as f:
                f.write('Name:       ' + name + '\n')
                f.write('Shape:      ' + str(array.shape) + '\n')
                f.write('Dtype:      ' + str(dtype) + '\n')
                f.write('Scale:      ' + str(quantization[0]) + '\n')
                f.write('Zero point: ' + str(quantization[1]) + '\n')
            if with_txt:
                if dtype in int_types:
                    fmt = '%+6d'
                elif dtype in float_types:
                    fmt = '%+1.5f'
                else:
                    raise TypeError('Invalid numpy dtype: {}'.format(dtype))
                array_sq = np.squeeze(array)
                if array_sq.ndim == 3:
                    array_sq = np.squeeze(array_sq[:, :, 0])
                elif array_sq.ndim == 4:
                    array_sq = np.squeeze(array_sq[:, :, 0, 0])
                elif array_sq.ndim == 5:
                    array_sq = np.squeeze(array_sq[:, :, 0, 0, 0])
                np.savetxt(main_name + '.txt', array_sq, fmt=fmt)
    else:
        tensor_names = []
        for tensor in tensor_list:
            if isinstance(tensor, str):
                tensor_names.append(tensor)
            if isinstance(tensor, dict):
                tensor_names.append(tensor['name'])
            elif isinstance(tensor, tf.Tensor):
                tensor_names.append(tensor.name)
        for td in tensor_details:
            name = td['name']
            if name in tensor_names:
                main_name = os.path.join(model_dir, name)
                os.makedirs(os.path.split(main_name)[0], exist_ok=True)
                quantization = td['quantization']
                index = td['index']

                array = interpreter.get_tensor(index)
                shape = array.shape
                dtype = array.dtype
                if len(shape) == 4:
                    array = np.transpose(array, [1, 2, 3, 0])
                array_bin = array.tobytes()

                with open(main_name + '.bin', mode='wb') as f:
                    f.write(array_bin)
                with open(main_name + '.info', mode='w') as f:
                    f.write('Name:       ' + name + '\n')
                    f.write('Shape:      ' + str(array.shape) + '\n')
                    f.write('Dtype:      ' + str(dtype) + '\n')
                    f.write('Scale:      ' + str(quantization[0]) + '\n')
                    f.write('Zero point: ' + str(quantization[1]) + '\n')
                if with_txt:
                    if dtype in int_types:
                        fmt = '%+6d'
                    elif dtype in float_types:
                        fmt = '%+1.5f'
                    else:
                        raise TypeError('Invalid numpy dtype: {}'.format(dtype))
                    array_sq = np.squeeze(array)
                    if array_sq.ndim == 3:
                        array_sq = np.squeeze(array_sq[:, :, 0])
                    elif array_sq.ndim == 4:
                        array_sq = np.squeeze(array_sq[:, :, 0, 0])
                    elif array_sq.ndim == 5:
                        array_sq = np.squeeze(array_sq[:, :, 0, 0, 0])
                    np.savetxt(main_name + '.txt', array_sq, fmt=fmt)
    print('Done.')


def write_quantization_params(model_file, model_file_quant, tensor_list=None, show_details=True):
    model_file = str(model_file)
    model_dir = model_file.replace('.tflite', '')
    model_file_quant = str(model_file_quant)
    model_dir_quant = model_file_quant.replace('.tflite', '')

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    interpreter_quant = tf.lite.Interpreter(model_path=model_file_quant)
    interpreter_quant.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    tensor_details_quant = interpreter_quant.get_tensor_details()

    if show_details:
        print('Tensor Details:')
        for detail in tensor_details:
            print(detail)
        print()
        print('Quantized Tensor Details:')
        for detail in tensor_details_quant:
            print(detail)
        print()

    print('Writing quantization information for {} ...'.format(os.path.split(model_file_quant)[1]), end=' ')
    if tensor_list is None:
        for i, tdq in enumerate(tensor_details_quant):
            name = tdq['name']
            if i >= len(tensor_details):
                break
            elif name.replace('_int8', '') != tensor_details[i]['name']:
                break
            else:
                main_name = os.path.join(model_dir, name.replace('_int8', ''))
                main_name_quant = os.path.join(model_dir_quant, name)
                with open(main_name + '.bin', mode='rb') as f:
                    arr_binary = f.read()
                with open(main_name + '.info', mode='r') as f:
                    lines = f.readlines()
                    shape = ast.literal_eval(lines[1][12:].rstrip())
                    dtype = lines[2][12:].rstrip()

                with open(main_name_quant + '.bin', mode='rb') as f:
                    arr_binary_quant = f.read()
                with open(main_name_quant + '.info', mode='r') as f:
                    lines = f.readlines()
                    dtype_quant = lines[2][12:].rstrip()
                    scale = float(lines[3][12:].rstrip())
                    offset = float(lines[4][12:].rstrip())

                if scale == offset == 0.0:
                    arr = np.frombuffer(arr_binary, dtype=dtype).reshape(shape).astype(np.float64)
                    arr_quant = np.frombuffer(arr_binary_quant, dtype=dtype_quant).reshape(shape).astype(np.float64)
                    quant_scale = (arr/arr_quant).astype(np.float32)
                    quant_scale = np.where(np.isinf(quant_scale),
                                           np.zeros(quant_scale.shape, dtype=np.float32), quant_scale)
                    dim = quant_scale.ndim
                    if dim == 1:  # Bias
                        quant_scale = quant_scale
                    elif dim == 4:  # Convolution
                        if quant_scale.shape[-1] == 1:  # Depthwise
                            quant_scale = np.mean(quant_scale, axis=(0, 1, 3))
                        else:
                            quant_scale = np.mean(quant_scale, axis=(0, 1, 2))
                    else:
                        raise(ValueError, 'Invalid tensor dimension: {}'.format(dim))
                    with open(main_name_quant + '.quant', mode='wb') as f:
                        f.write(np.reshape(quant_scale, (np.prod(quant_scale.shape))).tobytes())
                else:
                    with open(main_name_quant + '.quant', mode='wb') as f:
                        f.write(np.array(scale).tobytes())
    else:
        tensor_names = []
        for tensor in tensor_list:
            if isinstance(tensor, str):
                tensor_names.append(tensor)
            if isinstance(tensor, dict):
                tensor_names.append(tensor['name'])
            elif isinstance(tensor, tf.Tensor):
                tensor_names.append(tensor.name)
        for i, tdq in enumerate(tensor_details_quant):
            name = tdq['name']
            if i >= len(tensor_details):
                break
            elif name.replace('_int8', '') != tensor_details[i]['name']:
                break
            elif name.replace('_int8', '') in tensor_names:
                main_name = os.path.join(model_dir, name.replace('_int8', ''))
                main_name_quant = os.path.join(model_dir_quant, name)
                with open(main_name + '.bin', mode='rb') as f:
                    arr_binary = f.read()
                with open(main_name + '.info', mode='r') as f:
                    lines = f.readlines()
                    shape = ast.literal_eval(lines[1][12:].rstrip())
                    dtype = lines[2][12:].rstrip()

                with open(main_name_quant + '.bin', mode='rb') as f:
                    arr_binary_quant = f.read()
                with open(main_name_quant + '.info', mode='r') as f:
                    lines = f.readlines()
                    dtype_quant = lines[2][12:].rstrip()
                    scale = float(lines[3][12:].rstrip())
                    offset = float(lines[4][12:].rstrip())

                if scale == offset == 0.0:
                    arr = np.frombuffer(arr_binary, dtype=dtype).reshape(shape).astype(np.float64)
                    arr_quant = np.frombuffer(arr_binary_quant, dtype=dtype_quant).reshape(shape).astype(np.float64)
                    quant_scale = (arr/arr_quant).astype(np.float32)
                    quant_scale = np.where(np.isinf(quant_scale),
                                           np.zeros(quant_scale.shape, dtype=np.float32), quant_scale)
                    dim = quant_scale.ndim
                    if dim == 1:  # Bias
                        quant_scale = quant_scale
                    elif dim == 4:  # Convolution
                        if quant_scale.shape[-1] == 1:  # Depthwise
                            quant_scale = np.mean(quant_scale, axis=(0, 1, 3))
                        else:
                            quant_scale = np.mean(quant_scale, axis=(0, 1, 2))
                    else:
                        raise (ValueError, 'Invalid tensor dimension: {}'.format(dim))
                    with open(main_name_quant + '.quant', mode='wb') as f:
                        f.write(np.reshape(quant_scale, (np.prod(quant_scale.shape))).tobytes())
                else:
                    with open(main_name_quant + '.quant', mode='wb') as f:
                        f.write(np.array(scale).tobytes())
