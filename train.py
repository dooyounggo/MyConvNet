import os
import shutil
import numpy as np
import tensorflow as tf
from dataset import DataSet
import utils
import matplotlib.pyplot as plt
from optimizers import MomentumOptimizer as Optimizer
from parameters import subset
from parameters import ConvNet
from parameters import Evaluator
from parameters import Parameters
from parameters import init_from_pretrained_model


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
print('')

# Load trainval set and split into train/val sets
image_dirs, label_dirs, class_names = subset.read_subset(Param.train_dir, shuffle=Param.d['shuffle'],
                                                         sample_size=Param.train_sample_size)
train_size = len(image_dirs)
if Param.val_dir is None:
    val_size = int(train_size*0.1)    # FIXME
    val_set = DataSet(image_dirs[:val_size], label_dirs[:val_size],
                      class_names, random=Param.d['augment_pred'], **Param.d)
    train_set = DataSet(image_dirs[val_size:], label_dirs[val_size:],
                        class_names, random=Param.d['augment_train'], **Param.d)
else:
    image_dirs_val, label_dirs_val, _ = subset.read_subset(Param.val_dir, shuffle=Param.d['shuffle'],
                                                           sample_size=Param.val_sample_size)
    val_set = DataSet(image_dirs_val, label_dirs_val, class_names, random=Param.d['augment_pred'], **Param.d)
    train_set = DataSet(image_dirs, label_dirs, class_names, random=Param.d['augment_train'], **Param.d)

# Data check
# train_set.data_statistics(verbose=True)     # Comment this if your dataset is too big.
weighting_method = Param.d['loss_weighting']
w = None
if weighting_method is not None:
    if isinstance(weighting_method, (list, tuple)):
        w = weighting_method
    elif weighting_method.lower() == 'balanced':
        w = train_set.balanced_weights

# image_mean = train_set.image_mean
image_mean = 0.5
print('Image mean: {}\n'.format(image_mean))
np.save(os.path.join(Param.save_dir, 'img_mean'), image_mean)    # save image mean
Param.d['image_mean'] = image_mean

fp = open(os.path.join(Param.save_dir, 'parameters.txt'), 'w')
for k, v in Param.d.items():
    fp.write('{}:\t{}\n'.format(k, v))
fp.close()

# Initialize
model = ConvNet(Param.d['input_size'], len(class_names), loss_weights=w, **Param.d)
if Param.d['init_from_pretrained_model']:
    init_from_pretrained_model(Param.pretrained_dir)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **Param.d)

train_results = optimizer.train(save_dir=Param.save_dir, transfer_dir=Param.transfer_dir,
                                details=True, verbose=True, show_each_step=False, **Param.d)

model.session.close()
