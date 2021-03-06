"""
Inference using trained networks.
"""

from classification.parameters import *


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
    test_set = DataSet(image_dirs, label_dirs, class_names=class_names, out_size=Param.d['image_size_test'],
                       task_type=DataSet.IMAGE_ONLY,
                       resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                       **Param.d)

    image_mean_file = os.path.join(Param.save_dir, 'img_mean.npy')
    if os.path.isfile(image_mean_file):
        image_mean = np.load(image_mean_file).astype(np.float32)    # load mean image
        Param.d['image_mean'] = image_mean
    Param.d['monte_carlo'] = False

    # Initialize
    model = ConvNet(Param.d['input_size'], test_set.num_classes, loss_weights=None, **Param.d)
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
    test_x, _, test_y_pred, _ = model.predict(test_set, verbose=True, **Param.d)

    utils.plot_class_results(test_x, test_y_pred, fault=None, shuffle=False, class_names=class_names,
                             save_dir=os.path.join(Param.save_dir, 'results_inference'), start_idx=idx_start)
    plt.show()

    model.session.close()
