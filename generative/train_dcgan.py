from generative.parameters_gan import *
import shutil


if __name__ == '__main__':
    Param = Parameters()
    if os.path.exists(Param.save_dir):  # Check existing data
        olddir = os.path.join(Param.save_dir, '_old')    # To store existing date in /_old
        if os.path.exists(olddir):
            shutil.rmtree(olddir)   # Delete older data
        filenames = os.listdir(Param.save_dir)
        os.makedirs(olddir)
        for filename in filenames:
            try:
                full_filename = os.path.join(Param.save_dir, filename)
                if os.path.isdir(full_filename):    # Move existing data
                    shutil.copytree(full_filename, os.path.join(olddir, filename))
                    shutil.rmtree(full_filename)
                else:
                    shutil.copy2(full_filename, olddir)
                    os.remove(full_filename)
            except Exception as err:
                print(err)
    else:
        os.makedirs(Param.save_dir)

    # Load trainval set and split into train/val sets
    image_dirs, label_dirs, class_names = read_subset(Param.train_dir, shuffle=Param.d['shuffle'],
                                                      sample_size=Param.train_sample_size)
    train_size = len(image_dirs)
    if Param.val_dir is None:
        if Param.val_sample_size is None:
            val_set = None
            train_set = DataSet(image_dirs, None, num_classes=Param.d['latent_vector_length'],
                                out_size=Param.d['image_size'], task_type=DataSet.IMAGE_ONLY,
                                resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                                **Param.d)
        else:
            val_set = DataSet(image_dirs[:Param.val_sample_size], None, num_classes=Param.d['latent_vector_length'],
                              out_size=Param.d['image_size_test'], task_type=DataSet.IMAGE_ONLY,
                              resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                              **Param.d)
            train_set = DataSet(image_dirs[Param.val_sample_size:], None, num_classes=Param.d['latent_vector_length'],
                                out_size=Param.d['image_size'], task_type=DataSet.IMAGE_ONLY,
                                resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                                **Param.d)
    else:
        image_dirs_val, label_dirs_val, _ = read_subset(Param.val_dir, shuffle=Param.d['shuffle'],
                                                        sample_size=Param.val_sample_size)
        val_set = DataSet(image_dirs_val, label_dirs_val, num_classes=Param.d['latent_vector_length'],
                          out_size=Param.d['image_size_test'], task_type=DataSet.IMAGE_ONLY,
                          resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                          **Param.d)
        train_set = DataSet(image_dirs, label_dirs, num_classes=Param.d['latent_vector_length'],
                            out_size=Param.d['image_size'], task_type=DataSet.IMAGE_ONLY,
                            resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                            **Param.d)

    # Data check
    image_mean = Param.d['image_mean']
    weighting_method = Param.d.get('loss_weighting', None)
    if isinstance(weighting_method, (list, tuple)):
        w = weighting_method
    else:
        w = None

    print('Image mean:', Param.d['image_mean'], '\n')
    np.save(os.path.join(Param.save_dir, 'img_mean'), Param.d['image_mean'])  # save image mean

    fp = open(os.path.join(Param.save_dir, 'parameters.txt'), 'w')
    for k, v in Param.d.items():
        fp.write('{}:\t{}\n'.format(k, v))
    fp.close()

    # Initialize
    model = ConvNet(Param.d['input_size'], train_set.num_classes, loss_weights=w, **Param.d)
    if Param.d['init_from_public_checkpoint']:
        init_from_checkpoint(Param.checkpoint_dir, load_moving_average=Param.d.get('load_moving_average', False))
    evaluator = Evaluator(to_return='return_y_true', score_name='Generator Loss')
    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **Param.d)

    train_results = optimizer.train(save_dir=Param.save_dir, transfer_dir=Param.transfer_dir,
                                    details=True, verbose=True, show_each_step=False, show_percentage=False, **Param.d)

    model.session.close()
