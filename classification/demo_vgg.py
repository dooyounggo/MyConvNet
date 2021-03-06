from classification.parameters_vgg import *
import csv


if __name__ == '__main__':
    Param = Parameters()
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
    model = ConvNet(Param.d['input_size'], test_set.num_classes, loss_weights=None, **Param.d)
    evaluator = Evaluator()
    init_from_checkpoint(Param.checkpoint_dir, load_moving_average=Param.d.get('load_moving_average', False))
    model.session.run(tf.global_variables_initializer())

    test_x, test_y_true, test_y_pred, _ = model.predict(test_set, verbose=True, **Param.d)
    test_score = evaluator.score(test_y_true, test_y_pred)

    print(evaluator.name + ': {:.4f}'.format(test_score))
    utils.plot_class_results(test_x, test_y_true, test_y_pred, fault=False, shuffle=False, class_names=class_names,
                             save_dir=os.path.join(Param.save_dir, 'results_test/images'), start_idx=idx_start)
    utils.plot_class_results(test_x, test_y_true, test_y_pred, fault=True, shuffle=False, class_names=class_names)

    gcam = model.features(test_set, model.gcam, **Param.d)[0][..., 0]
    cmap = plt.get_cmap('gnuplot2')
    gcam = cmap(gcam)[..., 0:3]
    gcam = np.clip(test_x + gcam, 0, 1)
    # gcam = test_x*gcam[..., np.newaxis]

    utils.plot_class_results(gcam, test_y_true, test_y_pred, fault=None, shuffle=False, class_names=class_names,
                             save_dir=os.path.join(Param.save_dir, 'results_test/grad-cams'), start_idx=idx_start)

    # cm = utils.plot_confusion_matrix(test_y_true, test_y_pred, class_names=class_names, normalize=False)
    # fp = open(os.path.join(Param.save_dir, 'confusion_matrix.csv'), 'w', encoding='utf-8', newline='')
    # wrt = csv.writer(fp)
    # wrt.writerow(['id'] + list(class_names))
    # for i, line in enumerate(cm):
    #     wrt.writerow([class_names[i]] + list(line))
    # fp.close()

    plt.show()

    model.session.close()
