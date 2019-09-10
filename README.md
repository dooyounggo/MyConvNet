# MyConvNet
  Deep learning using TensorFlow low-level APIs

  Build your own convolutional neural networks using TensorFlow.
  
  Currently supports image classification and segmentation tasks.
  
  The code was verified using PyCharm.

## How-to
- Download all the files.
- Preprocess your data using scripts in subsets/.
- Edit parameters.py to change the dataset, model, directories, etc...
- Run train.py to train the model.
- Run test.py to test the trained model.

### Packages
- Tensorflow-gpu: 1.14.0
- Matplotlib: 3.1.0
- scikit-image: 0.15.0
- opencv-python: 4.1.0.25

### TODO
- Speedup: Training is roughly 2-3 times slower than tf_cnn_benchmark presumably due to CPU <-> GPU data transfer time.
