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
# node_names = ('block1/res0/conv2/bn', 'block2/res0/conv2/bn', 'block3/res0/conv2/bn', 'block4/res0/conv2/bn')
node_names = ('block1/res0/conv2/bn', 'block1/att0/att_mask', 'block2/res0/conv2/bn',
              'block2/att0/att_mask', 'block3/res0/conv2/bn', 'block3/att0/att_mask')
num_rows = 5
num_cols = 5
sample_size = 4

# Load test set
X_test, y_test, class_names = subset.read_subset(Param.test_dir, sample_size=sample_size, shuffle=True)
test_set = DataSet(X_test, y_test)

# Sanity check
image_mean = np.load(os.path.join(Param.save_dir, 'img_mean.npy')).astype(np.float32)    # load mean image
Param.d['image_mean'] = image_mean

# Initialize
model = ConvNet(Param.d['input_size'], len(class_names), y_test, **Param.d)
saver = tf.train.Saver()

if model_to_load is None:
    ckpt_to_load = tf.train.latest_checkpoint(Param.save_dir)
elif isinstance(model_to_load, str):
    ckpt_to_load = os.path.join(Param.save_dir, model_to_load)
else:
    fp = open(os.path.join(Param.save_dir, 'checkpoints.txt'), 'r')
    ckpt_list = fp.readlines()
    fp.close()
    ckpt_to_load = ckpt_list[model_to_load][:-1]

saver.restore(model.session, ckpt_to_load)    # restore learned weights
images, y_pred = model.session.run([model.X_all, model.pred], feed_dict={model.X_in: X_test,
                                                                         model.Y_in: y_test,
                                                                         model.is_train: False,
                                                                         model.monte_carlo: False,
                                                                         model.augmentation: False})
images = images + image_mean

num_gpus = Param.d.get('num_gpus', 1)
for i in range(sample_size):
    utils.imshow(images[i], y_test[i], y_pred[i], class_names=class_names, figure_title='Input image {}'.format(i))

for node in node_names:
    feat_tensors = [model.dicts[i][node] for i in range(num_gpus)]
    feat = model.session.run(feat_tensors, feed_dict={model.X_in: X_test,
                                                      model.Y_in: y_test,
                                                      model.is_train: False,
                                                      model.monte_carlo: False,
                                                      model.augmentation: False})
    features = np.concatenate(feat, axis=0)

    for i in range(sample_size):
        utils.show_features(features[i], group_features=True, num_rows=num_rows, num_cols=num_cols,
                            figure_title='Image {} '.format(i) + node)

plt.show()
