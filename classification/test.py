from classification.parameters import *
import csv


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
                   resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                   **Param.d)

image_mean = np.load(os.path.join(Param.save_dir, 'img_mean.npy')).astype(np.float32)    # load image mean
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

utils.plot_class_results(test_x, test_y_true, test_y_pred, fault=False, shuffle=False, class_names=class_names,
                         save_dir=os.path.join(Param.save_dir, 'results_test/images'), start_idx=idx_start)
utils.plot_class_results(test_x, test_y_true, test_y_pred, fault=True, shuffle=False, class_names=class_names)

gcam = model.features(test_set, model.gcam, **Param.d)[0][..., 0]
cmap = plt.get_cmap('gnuplot2')
gcam = cmap(gcam)[..., 0:3]
gcam = np.clip(test_x + gcam, 0, 1)
# gcam = test_x*gcam[..., np.newaxis]

utils.plot_class_results(gcam, test_y_true, test_y_pred, fault=None, shuffle=False, class_names=class_names,
                         save_dir=os.path.join(Param.save_dir, 'results_test/grad-cams'), start_idx=idx_start)

cm = utils.plot_confusion_matrix(test_y_true, test_y_pred, class_names=class_names, normalize=False)

fp = open(os.path.join(Param.save_dir, 'confusion_matrix.csv'), 'w', encoding='utf-8', newline='')
wrt = csv.writer(fp)
wrt.writerow(['id'] + list(class_names))
for i, line in enumerate(cm):
    wrt.writerow([class_names[i]] + list(line))
fp.close()

plt.show()

model.session.close()
