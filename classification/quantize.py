from classification.parameters import *
import quantization
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-training integer quantization',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--overwrite', '-o', help='Whether to overwrite existing tflite models',
                        type=str, default='False', metavar='')
    parser.add_argument('--saved_model', '-s', help='Whether to create saved_model from ckpt',
                        type=str, default='True', metavar='')
    parser.add_argument('--evaluate_models', '-e', help='Whether to evaluate tflite models',
                        type=str, default='True', metavar='')
    parser.add_argument('--write_tensors', '-w', help='Whether to write tensors in binary files',
                        type=str, default='True', metavar='')
    parser.add_argument('--num_repr_data', '-n', '--num_data', '--num_images', help='Number of representative images',
                        type=str, default='1000', metavar='')

    args, _ = parser.parse_known_args()
    overwrite = args.overwrite
    if overwrite.lower().strip() == 'true' or overwrite.strip() == '1':
        overwrite = True
    else:
        overwrite = False
    saved_model = args.saved_model
    if saved_model.lower().strip() == 'false' or saved_model.strip() == '0':
        saved_model = False
    else:
        saved_model = True
    evaluate_models = args.evaluate_models
    if evaluate_models.lower().strip() == 'false' or evaluate_models.strip() == '0':
        evaluate_models = False
    else:
        evaluate_models = True
    write_tensors = args.write_tensors
    if write_tensors.lower().strip() == 'false' or write_tensors.strip() == '0':
        write_tensors = False
    else:
        write_tensors = True
    num_repr_data = int(args.num_repr_data)

    Param = Parameters(parser=parser)
    model_to_load = Param.d['model_to_load']
    idx_start = 0
    idx_end = 50000

    # Load test set
    image_dirs, label_dirs, class_names = read_subset(Param.test_dir, shuffle=False, sample_size=Param.test_sample_size)
    num_data = len(image_dirs)
    image_dirs = image_dirs[idx_start:min(num_data, idx_end)]
    if label_dirs is not None:
        label_dirs = label_dirs[idx_start:min(num_data, idx_end)]
    Param.d['shuffle'] = False
    test_set = DataSet(image_dirs, label_dirs, class_names=class_names, out_size=Param.d['image_size_test'],
                       task_type=DataSet.IMAGE_CLASSIFICATION,
                       resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                       **Param.d)

    image_mean_file = os.path.join(Param.save_dir, 'img_mean.npy')
    if os.path.isfile(image_mean_file):
        image_mean = np.load(image_mean_file).astype(np.float32)    # load mean image
        Param.d['image_mean'] = image_mean
    Param.d['monte_carlo'] = False

    # Initialize
    Param.d['half_precision'] = False
    Param.d['channel_first'] = False
    Param.d['initial_drop_rate'] = 0.0
    Param.d['final_drop_rate'] = 0.0
    model = ConvNet(Param.d['input_size'], test_set.num_classes, loss_weights=None, **Param.d)
    evaluator = Evaluator()

    if model_to_load is None:
        ckpt_to_load = tf.train.latest_checkpoint(Param.save_dir)
    elif isinstance(model_to_load, str):
        ckpt_to_load = os.path.join(Param.save_dir, model_to_load)
    else:
        fp = open(os.path.join(Param.save_dir, 'checkpoints.txt'), 'r')
        ckpt_list = fp.readlines()
        fp.close()
        ckpt_to_load = os.path.join(Param.save_dir, ckpt_list[model_to_load].rstrip())

    image_dirs, label_dirs, class_names = read_subset(Param.train_dir, shuffle=True, sample_size=num_repr_data)
    train_set = DataSet(image_dirs, label_dirs, class_names=class_names,
                        out_size=Param.d['image_size'], task_type=DataSet.IMAGE_CLASSIFICATION,
                        resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                        **Param.d)
    images = np.empty([len(image_dirs)] + list(Param.d['image_size']), dtype=np.float32)
    print('Loading representative images...', end=' ')
    for i, (idir, ldir) in enumerate(zip(image_dirs, label_dirs)):
        img, _ = train_set._load_function(idir, ldir)
        images[i] = img
    print('Done.')
    print('')

    (tflite_model_file, tflite_model_quant_file) = quantization.quantize(model, images, ckpt_to_load, Param.save_dir,
                                                                         overwrite=overwrite, saved_model=saved_model,
                                                                         **Param.d)
    if evaluate_models:
        quantization.evaluate_quantized_model(tflite_model_file, tflite_model_quant_file,
                                              test_set, evaluator, num_processes=Param.d.get('num_parallel_calls', 4),
                                              **Param.d)
    if write_tensors:
        quantization.write_tensors(tflite_model_file, images[0], tensor_list=None, with_txt=True)
        quantization.write_tensors(tflite_model_quant_file, images[0], tensor_list=None, with_txt=True)
        quantization.write_quantization_params(tflite_model_file, tflite_model_quant_file, tensor_list=None,
                                               show_details=True)
