from segmentation.parameters_seg import *
import shutil


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
    val_size = int(train_size*0.1) if Param.val_sample_size is None else Param.val_sample_size
    val_set = DataSet(image_dirs[:val_size], label_dirs[:val_size], class_names=class_names,
                      out_size=Param.d['image_size_test'], task_type=DataSet.IMAGE_SEGMENTATION,
                      resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                      **Param.d)
    train_set = DataSet(image_dirs[val_size:], label_dirs[val_size:], class_names=class_names,
                        out_size=Param.d['image_size'], task_type=DataSet.IMAGE_SEGMENTATION,
                        resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                        **Param.d)
else:
    image_dirs_val, label_dirs_val, _ = read_subset(Param.val_dir, shuffle=Param.d['shuffle'],
                                                    sample_size=Param.val_sample_size)
    val_set = DataSet(image_dirs_val, label_dirs_val, class_names=class_names,
                      out_size=Param.d['image_size_test'], task_type=DataSet.IMAGE_SEGMENTATION,
                      resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                      **Param.d)
    train_set = DataSet(image_dirs, label_dirs, class_names=class_names,
                        out_size=Param.d['image_size'], task_type=DataSet.IMAGE_SEGMENTATION,
                        resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                        **Param.d)

# Data check
image_mean = Param.d['image_mean']
weighting_method = Param.d.get('loss_weighting', None)
if weighting_method is not None:
    if isinstance(weighting_method, (list, tuple)):
        w = weighting_method
    elif weighting_method.lower() == 'balanced':
        train_set.data_statistics(verbose=True)
        w = train_set.balanced_weights
        if image_mean is None:
            Param.d['image_mean'] = train_set.image_mean
    else:
        w = None
else:
    w = None
    if image_mean is None:
        train_set.data_statistics(verbose=True)
        Param.d['image_mean'] = train_set.image_mean

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
evaluator = Evaluator(num_classes=train_set.num_classes)
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **Param.d)

train_results = optimizer.train(save_dir=Param.save_dir, transfer_dir=Param.transfer_dir,
                                details=True, verbose=True, show_each_step=False, **Param.d)

model.session.close()
