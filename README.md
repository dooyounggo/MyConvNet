# MyConvNet
  Deep learning using TensorFlow low-level APIs

  Build your own convolutional neural networks using TensorFlow.
  
  Supports image classification and segmentation tasks.
  
  The code was verified using PyCharm on Windows.

## How To Run
- Download all the files.
- Preprocess your data using scripts in subsets/.
- Build your own networks by modifying scripts in models/.
- Edit parameters.py to change the dataset, model, directories, etc...
- Run train.py to train the model.
- Run test.py to test the trained model.

### Packages
- tensorflow-gpu: 1.14.0
- numpy: 1.16.4
- matplotlib: 3.1.0
- scikit-image: 0.15.0
- scikit-learn: 0.21.2
- opencv-python: 4.1.0.25

### TODO
- Speedup: Training is roughly 2-3 times slower than tf_cnn_benchmark presumably due to CPU <-> GPU data transfer time.
- Include detection task
