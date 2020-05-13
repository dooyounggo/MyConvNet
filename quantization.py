"""
Model quantization.
Note that only the float32 data type and the "NHWC" format are supported.
"""

import os
import sys
import pathlib
import time
import pydot
import numpy as np
import tensorflow.compat.v1 as tf
import multiprocessing as mp
from subsets.subset_functions import resize_with_crop_or_pad


def quantize(model, images, ckpt_dir, save_dir, overwrite=True, **kwargs):
    """
    :param model: ConvNet, a model to be quantized.
    :param images: np.ndarray, representative images used for quantization.
    :param ckpt_dir: string, a path to saved checkpoint.
    :param save_dir: string, a path to save models.
    :param overwrite: bool, whether to overwrite tflite files already exist.
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
        if model.channel_first:
            model.X = tf.transpose(input_tensor, perm=[0, 3, 1, 2])
        else:
            model.X = input_tensor
        if model.dtype is not tf.float32:
            model.X = tf.cast(model.X, dtype=model.dtype)
        d = model._build_model(**kwargs)
        output_tensor = d['pred']
    if kwargs.get('zero_center', True):
        image_mean = kwargs.get('image_mean', 0.5)
    else:
        image_mean = 0.0
    scale_factor = kwargs.get('scale_factor', 2.0)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess, ckpt_dir)

    tflite_models_dir = pathlib.Path(os.path.join(save_dir, 'tflite'))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/'model.tflite'
    tflite_model_quant_file = tflite_models_dir/'model_quantized.tflite'

    tflite_graphviz_dir = tflite_models_dir/'graphviz'
    tflite_graphviz_quant_dir = tflite_models_dir/'graphviz_quant'
    tflite_graphviz_dir.mkdir(exist_ok=True, parents=True)
    tflite_graphviz_quant_dir.mkdir(exist_ok=True, parents=True)
    dotfile_names = ['toco_AT_IMPORT', 'toco_AFTER_TRANSFORMATIONS', 'toco_AFTER_ALLOCATION']

    if overwrite or not tflite_model_file.exists():
        converter = tf.lite.TFLiteConverter.from_session(sess=sess,
                                                         input_tensors=[input_tensor],
                                                         output_tensors=[output_tensor])
        print('Converting the model...')
        converter.dump_graphviz_dir = str(tflite_graphviz_dir)
        tflite_model = converter.convert()
        tflite_model_file.write_bytes(tflite_model)
        if os.name != 'nt':  # Conversion from dot to svg is not supported on Windows due to UnicodeDecodeError.
            for dotname in dotfile_names:
                (dotgraph,) = pydot.graph_from_dot_file(os.path.join(str(tflite_graphviz_dir), dotname + '.dot'))
                dotgraph.write_svg(os.path.join(str(tflite_graphviz_dir), dotname + '.svg'))
        print('Done. \n')

    if overwrite or not tflite_model_quant_file.exists():
        converter_quant = tf.lite.TFLiteConverter.from_session(sess=sess,
                                                               input_tensors=[input_tensor],
                                                               output_tensors=[output_tensor])
        converter_quant.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

        def repr_data_gen():
            for img in images:
                yield [(img[np.newaxis, ...] - image_mean)*scale_factor]

        converter_quant.representative_dataset = repr_data_gen
        converter_quant.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_quant.inference_input_type = tf.uint8
        converter_quant.inference_output_type = tf.uint8

        print('Converting the quantized model...')
        converter_quant.dump_graphviz_dir = str(tflite_graphviz_quant_dir)
        tflite_model_quant = converter_quant.convert()
        tflite_model_quant_file.write_bytes(tflite_model_quant)
        if os.name != 'nt':
            for dotname in dotfile_names:
                (dotgraph,) = pydot.graph_from_dot_file(os.path.join(str(tflite_graphviz_quant_dir), dotname + '.dot'))
                dotgraph.write_svg(os.path.join(str(tflite_graphviz_quant_dir), dotname + '.svg'))
        print('Done. \n')

    return tflite_model_file, tflite_model_quant_file


def evaluate_quantized_model(model_file, model_quant_file, test_set, evaluator, show_details=False, **kwargs):
    dataset = tf.data.Dataset.from_tensor_slices((test_set.image_dirs, test_set.label_dirs))
    if not test_set.from_memory:
        dataset = dataset.map(lambda image_dir, label_dir: tuple(tf.py_func(test_set._load_function,
                                                                            (image_dir, label_dir),
                                                                            (tf.float32, tf.float32))),
                              num_parallel_calls=1)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    sess = tf.Session()
    sess.run(iterator.initializer)
    input_image_tensor, input_label_tensor = iterator.get_next()

    interpreter = tf.lite.Interpreter(model_path=str(model_file))
    interpreter.allocate_tensors()
    interpreter_quant = tf.lite.Interpreter(model_path=str(model_quant_file))
    interpreter_quant.allocate_tensors()

    if show_details:
        tensor_details = interpreter.get_tensor_details()
        print('Tensor Details:')
        for detail in tensor_details:
            print(detail)
        print('')
        tensor_details = interpreter_quant.get_tensor_details()
        print('Quantized Tensor Details:')
        for detail in tensor_details:
            print(detail)
        print('')

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_details_quant = interpreter_quant.get_input_details()[0]
    output_details_quant = interpreter_quant.get_output_details()[0]
    input_index_quant = input_details_quant['index']
    output_index_quant = output_details_quant['index']

    image_mean = kwargs.get('image_mean', 0.5)
    scale_factor = kwargs.get('scale_factor', 2.0)

    label_shape = [test_set.num_examples] + list(output_details_quant['shape'][1:])
    gt_label = np.empty(label_shape[:-1], dtype=np.float32)
    results = np.empty(label_shape, dtype=np.float32)
    results_quant = np.empty(label_shape, dtype=output_details_quant['dtype'])

    total_time = 0
    for i in range(test_set.num_examples):
        if (i % 100) == 0:
            print('Evaluating models... {:5d}/{}'.format(i, test_set.num_examples))
        t_start = time.time()
        input_image, input_label = sess.run([input_image_tensor, input_label_tensor])
        input_image = resize_with_crop_or_pad(input_image[0],
                                              out_size=input_details_quant['shape'][1:])
        input_image = (input_image[np.newaxis, ...] - image_mean)*scale_factor

        if i < 10:
            t = time.time()
            print('Invoking the model...', end=' ')
        interpreter.set_tensor(input_index, input_image)
        interpreter.invoke()
        if i < 10:
            print('Done. Elapsed time: {} sec.'.format(time.time() - t))

        input_quant_details = input_details_quant['quantization']
        if input_quant_details[0] > 0.0:
            input_image_quant = input_image/input_quant_details[0] + input_quant_details[1]
        else:
            input_image_quant = input_image
        input_image_quant = input_image_quant.astype(input_details_quant['dtype'])

        if i < 10:
            t = time.time()
            print('Invoking the quantized model...', end=' ')
        interpreter_quant.set_tensor(input_index_quant, input_image_quant)
        interpreter_quant.invoke()
        if i < 10:
            print('Done. Elapsed time: {} sec.'.format(time.time() - t))

        gt_label[i] = input_label[0]
        results[i] = interpreter.get_tensor(output_index)[0]
        results_quant[i] = interpreter_quant.get_tensor(output_index_quant)[0]

        if i < 10:
            total_time += time.time() - t_start
            print('Estimated test time: {} min.'.format(int(total_time/(i + 1)*test_set.num_examples/60)))
        if i < 100:
            print('{}. GT:'.format(i))
            print(gt_label[i].astype(int))
            print('{}. Before:'.format(i))
            print(np.argmax(results[i], axis=-1))
            print('{}. After:'.format(i))
            print(np.argmax(results_quant[i], axis=-1))
            print('')

    results_argmax = np.argmax(results, axis=-1)
    results_quant_argmax = np.argmax(results_quant, axis=-1)
    accuracy = evaluator.score(gt_label[..., np.newaxis],
                               results_argmax[..., np.newaxis])
    accuracy_quant = evaluator.score(gt_label[..., np.newaxis],
                                     results_quant_argmax[..., np.newaxis])
    is_different = np.not_equal(results_argmax, results_quant_argmax)

    print('\nAccuracy Before Quantization: {:.4f}'.format(accuracy))
    print('Accuracy After Quantization:  {:.4f}'.format(accuracy_quant))
    print('Number of Different Results: {}/{}'.format(np.sum(is_different), np.prod(is_different.shape)))


def evaluate_quantized_model_multiprocess(model_file, model_quant_file, test_set, evaluator, show_details=False,
                                          num_processes=4, **kwargs):
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

    if show_details:
        tensor_details = interpreter.get_tensor_details()
        print('Tensor Details:')
        for detail in tensor_details:
            print(detail)
        print('')
        tensor_details = interpreter_quant.get_tensor_details()
        print('Quantized Tensor Details:')
        for detail in tensor_details:
            print(detail)
        print('')

    input_details_quant = interpreter_quant.get_input_details()[0]
    output_details_quant = interpreter_quant.get_output_details()[0]

    image_mean = kwargs.get('image_mean', 0.5)
    scale_factor = kwargs.get('scale_factor', 2.0)

    params = dict()
    params['model_file'] = model_file
    params['model_quant_file'] = model_quant_file
    params['num_images'] = test_set.num_examples
    params['image_mean'] = image_mean
    params['scale_factor'] = scale_factor
    params['num_processes'] = num_processes

    label_shape = [test_set.num_examples] + list(output_details_quant['shape'][1:])
    idx = mp.Value('l', lock=False)
    image_arr = mp.Array('f', int(np.prod(input_details_quant['shape'][1:])), lock=False)
    gt_label_arr = mp.Array('f', int(np.prod(label_shape[:-1])), lock=False)
    results_arr = mp.Array('f', int(np.prod(label_shape)), lock=False)
    results_quant_arr = mp.Array('f', int(np.prod(label_shape)), lock=False)
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

    gt_label = np.empty(label_shape[:-1], dtype=np.float32)
    results = np.empty(label_shape, dtype=np.float32)
    results_quant = np.empty(label_shape, dtype=np.float32)
    gt_len = np.prod(label_shape[1:-1]).astype(int)
    r_len = np.prod(label_shape[1:]).astype(int)
    for i in range(test_set.num_examples):
        gt_label[i] = np.array(gt_label_arr[i*gt_len:(i + 1)*gt_len]).reshape(label_shape[1:-1])
        results[i] = np.array(results_arr[i*r_len:(i + 1)*r_len]).reshape(label_shape[1:])
        results_quant[i] = np.array(results_quant_arr[i*r_len:(i + 1)*r_len]).reshape(label_shape[1:])

    results_argmax = np.argmax(results, axis=-1)
    results_quant_argmax = np.argmax(results_quant, axis=-1)
    accuracy = evaluator.score(gt_label[..., np.newaxis],
                               results_argmax[..., np.newaxis])
    accuracy_quant = evaluator.score(gt_label[..., np.newaxis],
                                     results_quant_argmax[..., np.newaxis])
    is_different = np.not_equal(results_argmax, results_quant_argmax)

    print('\nAccuracy Before Quantization: {:.4f}'.format(accuracy))
    print('Accuracy After Quantization:  {:.4f}'.format(accuracy_quant))
    print('Number of Different Results: {}/{}'.format(np.sum(is_different), np.prod(is_different.shape)))


def load_function(idx, image, gt_label, w_lock, r_lock, session, iterator, interpreter, **kwargs):
    num_images = kwargs['num_images']
    image_mean = kwargs['image_mean']
    scale_factor = kwargs['scale_factor']

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
            image_np[:] = input_image.reshape(np.prod(image_shape).astype(int)).copy()
            label_np[i*label_len:(i + 1)*label_len] = input_label.reshape(label_len).copy()
            r_lock.release()

            if i < 100:
                print('{}. GT:'.format(i))
                print(input_label.astype(int))


def tflite_process(idx, image, results, results_quant, w_lock, r_lock, **kwargs):
    model_file = kwargs['model_file']
    model_quant_file = kwargs['model_quant_file']
    num_images = kwargs['num_images']
    num_processes = kwargs['num_processes']

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
    label_len = np.prod(interpreter_quant.get_output_details()[0]['shape'][1:]).astype(int)
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
            if (i % 100) == 0:
                print('Evaluating models... {:5d}/{}'.format(i, num_images))
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
            results_np[i*label_len:(i + 1)*label_len] = output.reshape(label_len).copy()
            results_quant_np[i*label_len:(i + 1)*label_len] = output_quant.reshape(label_len).astype(np.float32).copy()

            i_local += 1
            if i < 100:
                total_time += time.time() - t_start
                print('Estimated test time: {} min.'.format(int(total_time/i_local*num_images/60/num_processes)))
                print('{}. Before:'.format(i))
                print(np.argmax(output, axis=-1))
                print('{}. After:'.format(i))
                print(np.argmax(output_quant, axis=-1))
                print('')
                sys.stdout.flush()
