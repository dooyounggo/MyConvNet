"""
Model quantization.
Note that only single GPU inference, the float32 data type, and the "NHWC" format are supported.
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
import pathlib
import time
from subsets.subset_functions import resize_with_crop_or_pad


def quantize(model, images, ckpt_dir, save_dir, **kwargs):
    """
    :param model: ConvNet, a model to be quantized.
    :param images: np.ndarray, representative images used for quantization.
    :param ckpt_dir: string, a path to saved checkpoint.
    :param save_dir: string, a path to save models.
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

    converter = tf.lite.TFLiteConverter.from_session(sess=sess,
                                                     input_tensors=[input_tensor],
                                                     output_tensors=[output_tensor])

    print('Converting the model.')
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path(os.path.join(save_dir, 'tflite'))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/'model.tflite'
    tflite_model_quant_file = tflite_models_dir/'model_quantized.tflite'
    tflite_model_file.write_bytes(tflite_model)

    # if not tflite_model_quant_file.exists():  # FIXME
    if True:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

        def repr_data_gen():
            for img in images:
                yield [(img[np.newaxis, ...] - image_mean)*scale_factor]

        converter.representative_dataset = repr_data_gen
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        print('Converting the quantized model.')
        tflite_model_quant = converter.convert()
        tflite_model_quant_file.write_bytes(tflite_model_quant)

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

    label_shape = [test_set.num_examples, test_set.num_classes]
    gt_label = np.empty(label_shape, dtype=np.float32)
    results = np.empty(label_shape, dtype=np.float32)
    results_quant = np.empty(label_shape, dtype=output_details_quant['dtype'])
    image_mean = kwargs.get('image_mean', 0.5)
    scale_factor = kwargs.get('scale_factor', 2.0)
    total_time = 0
    input_image_tensor, input_label_tensor = iterator.get_next()
    for i in range(test_set.num_examples):
        if (i % 100) == 0:
            print('Evaluating models... {:5d}/{}'.format(i, test_set.num_examples))
        if i < 10:
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
    accuracy = evaluator.score(gt_label, results)
    accuracy_quant = evaluator.score(gt_label, results_quant)

    print('Accuracy Before Quantization: {:.2f}'.format(accuracy))
    print('Accuracy After Quantization:  {:.2f}'.format(accuracy_quant))
