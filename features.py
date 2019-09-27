"""
Deep feature visualization.
"""

import os
import numpy as np
import tensorflow as tf
from dataset import DataSet
import utils
import matplotlib.pyplot as plt
from parameters import *


Param = Parameters()
model_to_load = Param.d['model_to_load']
feature_names = ('block_1', 'block_2', 'block_3', 'block_4')
num_rows = 5
num_cols = 5
sample_size = 4

# Load test set
image_dirs, label_dirs, class_names = subset.read_subset(Param.test_dir, shuffle=True, sample_size=sample_size)
Param.d['num_gpus'] = 1  # Utilizes only one GPU for convenience
Param.d['shuffle'] = False
test_set = DataSet(image_dirs, label_dirs, class_names, **Param.d)

image_mean = np.load(os.path.join(Param.save_dir, 'img_mean.npy')).astype(np.float32)    # load mean image
Param.d['image_mean'] = image_mean
Param.d['monte_carlo'] = False
Param.d['channel_first'] = False

# Initialize
model = ConvNet(Param.d['input_size'], len(class_names), loss_weights=None, **Param.d)
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

utils.plot_class_results(test_x, test_y_true, test_y_pred, fault=None, shuffle=False, class_names=class_names)

tensors = [model.d[name] for name in feature_names]
features = model.features(test_set, tensors, **Param.d)

for i, feat in enumerate(features):
    for n in range(sample_size):
        utils.show_features(feat[n], group_features=True, num_rows=num_rows, num_cols=num_cols,
                            figure_title='Image {} '.format(n) + feature_names[i - 1])
plt.show()
