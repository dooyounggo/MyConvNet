from generative.parameters_gan import *


if __name__ == '__main__':
    Param = Parameters()
    model_to_load = Param.d['model_to_load']

    # Generate test set
    num_examples = Param.test_sample_size if Param.test_sample_size is not None else 100

    image_size = Param.d['image_size_test'] if Param.d['image_size_test'] is not None else Param.d['image_size']
    image_dirs = np.zeros([num_examples] + list(image_size), dtype=np.float32)
    label_dirs = np.random.uniform(-1.0, 1.0, [num_examples, Param.d['latent_vector_length']]).astype(np.float32)

    Param.d['shuffle'] = False
    test_set = DataSet(image_dirs, label_dirs, num_classes=Param.d['latent_vector_length'],
                       out_size=Param.d['image_size_test'],
                       task_type=DataSet.IMAGE_CLASSIFICATION,
                       resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                       from_memory=True,  # Data is numpy arrays
                       **Param.d)

    image_mean_file = os.path.join(Param.save_dir, 'img_mean.npy')
    if os.path.isfile(image_mean_file):
        image_mean = np.load(image_mean_file).astype(np.float32)    # load mean image
        Param.d['image_mean'] = image_mean

    # Initialize
    model = ConvNet(Param.d['input_size'], test_set.num_classes, loss_weights=None, **Param.d)
    evaluator = Evaluator(to_return='return_y_true', score_name='Generator Loss')
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
    _, test_y_true, test_y_pred, _ = model.predict(test_set, verbose=True, **Param.d)
    test_score = evaluator.score(test_y_true, test_y_pred)

    print(evaluator.name + ': {:.4f}'.format(test_score))

    test_y_pred = test_y_pred

    fake_label = np.zeros([test_y_true.shape[0], 2])
    fake_label[:, 0] = 1
    utils.plot_class_results(test_y_pred, fake_label, fault=None, shuffle=False, class_names=('fake', 'real'))
    plt.show()

    model.session.close()
