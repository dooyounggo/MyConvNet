import os
import numpy as np
import tensorflow as tf
from dataset import DataSet
import utils
import matplotlib.pyplot as plt
from parameters import subset
from parameters import ConvNet
from parameters import Parameters


Param = Parameters()
model_to_load = Param.d['model_to_load']

# Load test set
image_dirs, label_dirs, class_names = subset.read_subset(Param.test_dir, shuffle=False,
                                                         sample_size=Param.test_sample_size)
Param.d['shuffle'] = False
test_set = DataSet(image_dirs, label_dirs, class_names, **Param.d)

image_mean = np.load(os.path.join(Param.save_dir, 'img_mean.npy')).astype(np.float32)    # load mean image
Param.d['image_mean'] = image_mean
Param.d['monte_carlo'] = False

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
test_x, _, test_y_pred, _ = model.predict(test_set, verbose=True, **Param.d)

utils.plot_class_results(test_x, test_y_pred, test_y_pred, fault=None, shuffle=False, class_names=class_names,
                         save_dir=os.path.join(Param.save_dir, 'results_inference'))
plt.show()

model.session.close()
