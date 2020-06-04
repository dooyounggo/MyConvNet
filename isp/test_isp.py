from isp.parameters_isp import *
import cv2


if __name__ == '__main__':
    Param = Parameters()
    model_to_load = Param.d['model_to_load']
    idx_start = 0
    idx_end = 10000

    # Load test set
    image_dirs, label_dirs, class_names = read_subset(Param.test_dir, shuffle=False, sample_size=Param.test_sample_size)
    num_data = len(image_dirs)
    image_dirs = image_dirs[idx_start:min(num_data, idx_end)]
    if label_dirs is not None:
        label_dirs = label_dirs[idx_start:min(num_data, idx_end)]
    Param.d['shuffle'] = False
    test_set = DataSet(image_dirs, label_dirs, out_size=Param.d['image_size_test'], task_type=DataSet.IMAGE_ONLY,
                       resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                       **Param.d)

    image_mean_file = os.path.join(Param.save_dir, 'img_mean.npy')
    if os.path.isfile(image_mean_file):
        image_mean = np.load().astype(np.float32)    # load mean image
        Param.d['image_mean'] = image_mean
    Param.d['monte_carlo'] = False

    # Initialize
    model = ConvNet(Param.d['input_size'], test_set.num_classes, loss_weights=None, **Param.d)
    evaluator = Evaluator()
    saver = tf.train.Saver()

    if model_to_load is None:
        ckpt_to_load = tf.train.latest_checkpoint(Param.save_dir)
    elif isinstance(model_to_load, str):
        ckpt_to_load = os.path.join(Param.save_dir, model_to_load)
    else:
        fp = open(os.path.join(Param.save_dir, 'checkpoints.txt'), 'r')
        ckpt_list = fp.readlines()
        fp.close()
        ckpt_to_load = os.path.join(Param.save_dir, ckpt_list[model_to_load].rstrip())

    saver.restore(model.session, ckpt_to_load)    # restore learned weights
    test_x, test_y_true, test_y_pred, _ = model.predict(test_set, verbose=True, **Param.d)
    test_score = evaluator.score(test_y_true, test_y_pred)

    print(evaluator.name + ': {:.4f}'.format(test_score))

    utils.imshow_subplot(test_x, num_rows=3, num_cols=3, figure_title='Noisy Images')
    utils.imshow_subplot(test_y_true, num_rows=3, num_cols=3, figure_title='GT Images')
    utils.imshow_subplot(test_y_pred, num_rows=3, num_cols=3, figure_title='Denoised Images')

    save_dir = os.path.join(Param.save_dir, 'results_test')
    for i, (x, y_t, y_p) in enumerate(zip(test_x, test_y_true, test_y_pred)):
        cv2.imwrite(os.path.join(save_dir, 'noisy', f'{i:5d}.jpg'), x, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(os.path.join(save_dir, 'ground_truth', f'{i:5d}.jpg'), y_t, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(os.path.join(save_dir, 'denoised', f'{i:5d}.jpg'), y_p, [cv2.IMWRITE_JPEG_QUALITY, 100])

    model.session.close()
