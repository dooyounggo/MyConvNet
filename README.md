# MyConvNet
  Deep learning using TensorFlow low-level APIs.

  Build your own convolutional neural networks using TensorFlow.
  
  Supports image classification and semantic segmentation tasks.
  
  The code was verified on Windows 10 and Ubuntu 18.04 using PyCharm with Anaconda.
  
  Use Linux for faster training.

## How To Run
- Download all the files.
- Prepare your data using scripts in subsets/.
- Build your own networks by modifying scripts in models/.
- Edit parameters.py to change the dataset, model, directories, etc...
- Run train.py to train the model.
- Run test.py to test the trained model.

### Instruction
- Check out the basic [instruction](https://www.dropbox.com/s/64wtb6kvn9ms5o3/MyConvNet.pptx?dl=0).

### Packages
- Python: 3.7
- tensorflow-gpu: 1.15.0 (cudatoolkit: 10.0, cudnn: 7.6.5)
- numpy: 1.17.4
- matplotlib: 3.1.1
- scikit-image: 0.15.0
- scikit-learn: 0.22
- opencv-python: 4.1.2.30

### Checkpoints (ImageNet)
| Model | Top-1 Acc. | Top-5 Acc. | Details | Params | ckpt |
|---|---|---|---|---|---|
| ResNet-v1.5-50 |  |  | Inception preprocessing | [params.py](https://www.dropbox.com/s/lhmnshgfs9jvrfd/imagenet_res50.py?dl=0) |  |

### TODO
- Speedup: Training is slower than tf_cnn_benchmark.
- Include the detection task, GAN, etc..
