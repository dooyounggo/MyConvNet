from classification.parameters import *
import quantization


num_repr_data = 1000
Param = Parameters()
model_to_load = Param.d['model_to_load']
idx_start = 0
idx_end = 50000

# Load test set
image_dirs, label_dirs, class_names = read_subset(Param.test_dir, shuffle=False, sample_size=Param.test_sample_size)
num_data = len(image_dirs)
image_dirs = image_dirs[idx_start:min(num_data, idx_end)]
if label_dirs is not None:
    label_dirs = label_dirs[idx_start:min(num_data, idx_end)]
Param.d['shuffle'] = False
test_set = DataSet(image_dirs, label_dirs, class_names=class_names, out_size=Param.d['image_size_test'],
                   task_type=DataSet.IMAGE_CLASSIFICATION,
                   resize_method=Param.d['resize_type_test'], resize_randomness=Param.d['resize_random_test'],
                   **Param.d)

image_mean = np.load(os.path.join(Param.save_dir, 'img_mean.npy')).astype(np.float32)    # load image mean
Param.d['image_mean'] = image_mean
Param.d['monte_carlo'] = False

# Initialize
Param.d['half_precision'] = False
Param.d['channel_first'] = False
model = ConvNet(Param.d['input_size'], test_set.num_classes, loss_weights=None, **Param.d)
evaluator = Evaluator()

if model_to_load is None:
    ckpt_to_load = tf.train.latest_checkpoint(Param.save_dir)
elif isinstance(model_to_load, str):
    ckpt_to_load = os.path.join(Param.save_dir, model_to_load)
else:
    fp = open(os.path.join(Param.save_dir, 'checkpoints.txt'), 'r')
    ckpt_list = fp.readlines()
    fp.close()
    ckpt_to_load = os.path.join(Param.save_dir, ckpt_list[model_to_load].rstrip())

image_dirs, label_dirs, class_names = read_subset(Param.train_dir, shuffle=True, sample_size=num_repr_data)
train_set = DataSet(image_dirs, label_dirs, class_names=class_names,
                    out_size=Param.d['image_size'], task_type=DataSet.IMAGE_CLASSIFICATION,
                    resize_method=Param.d['resize_type'], resize_randomness=Param.d['resize_random'],
                    **Param.d)
images = np.empty([len(image_dirs)] + list(Param.d['image_size']), dtype=np.float32)
print('Loading representative images...', end=' ')
for i, (idir, ldir) in enumerate(zip(image_dirs, label_dirs)):
    img, _ = train_set._load_function(idir, ldir)
    images[i] = img
print('Done.')
print('')

(tflite_model_file, tflite_model_quant_file) = quantization.quantize(model, images, ckpt_to_load, Param.save_dir)
quantization.evaluate_quantized_model(tflite_model_file, tflite_model_quant_file, test_set, evaluator,
                                      show_details=True, **Param.d)
