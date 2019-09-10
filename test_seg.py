import os
import numpy as np
import tensorflow as tf
from dataset import DataSet
import utils
import matplotlib.pyplot as plt
from parameters_seg import subset
from parameters_seg import ConvNet
from parameters_seg import Evaluator
from parameters_seg import Parameters


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
test_x += image_mean
test_score = evaluator.score(test_y_true, test_y_pred)

print(evaluator.name + ': {:.4f}'.format(test_score))

utils.plot_seg_results(test_x, test_y_true, test_y_pred, save_dir=os.path.join(Param.save_dir, 'results'))
plt.show()

model.session.close()
