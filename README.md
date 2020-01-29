# MyConvNet
  Deep learning using TensorFlow low-level APIs.

  Build your own convolutional neural networks using TensorFlow.
  
  Supports image classification and semantic segmentation tasks.
  
  Verified on Windows 10 and Ubuntu 18.04 using PyCharm with Anaconda.
  
  Check out the [instruction](https://www.dropbox.com/s/64wtb6kvn9ms5o3/MyConvNet.pptx?dl=0).

## How To Run
- Download all the files.
- Prepare your data using scripts in subsets/.
- Build your own networks by modifying scripts in models/.
- Edit parameters.py to change the dataset, model, directories, etc...
- Run train.py to train the model.
- Run test.py to test the trained model.

### Notes
- In the segmentation task, pixels with a value of 0 are ignored, so assign 1 to the first class.
- Use Linux for faster training.
- Multi-GPU training is available based on the parameter server strategy.
- For NCCL-based distributed training, use nccl/convnet.py and nccl/optimizer.py (experimental, available on Linux).
- Batch statistics of multiple devices are updated successively.
- Check out [REFERENCES.md](https://github.com/dooyounggo/MyConvNet/blob/master/REFERENCES.md) for papers and code examples.

### Packages
- Python: 3.7
- tensorflow-gpu: 1.14.0 or 1.15.0 (cudatoolkit: 10.0, cudnn: 7.6.5)
- numpy: 1.17.4
- matplotlib: 3.1.1
- scikit-image: 0.15.0
- scikit-learn: 0.22
- opencv-python: 4.1.2.30

### Checkpoints (ImageNet - subtracted by 0.5 and multiplied by 2, ranging in [-1.0, 1.0])
| Model | Top-1 Acc | Top-5 Acc | Details | Params | Ckpt |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-v1.5-50 | [76.35%](https://www.dropbox.com/s/4aoscqqovpdaqwr/ResNet-v1.5-50_ImageNet.svg?dl=0) | 92.94% | Inception preprocessing | [#.py](https://www.dropbox.com/s/lhmnshgfs9jvrfd/imagenet_res50.py?dl=0) | [#.zip](https://www.dropbox.com/s/ruxx6lhmkzmu7u9/ResNet-v1.5-50_ImageNet.zip?dl=0) |
| ResNet-v1.5-50 | [76.50%](https://www.dropbox.com/s/1h8udkqxi97fhg4/learning_curve-result-1.svg?dl=0) | 93.06% | + 30 epochs (120 in total) | [#.py](https://www.dropbox.com/s/w197etq5hkl4koy/ResNet-v1.5-50_ImageNet.py?dl=0) | [#.zip](https://www.dropbox.com/s/xl15y6g0n4aaq20/ResNet-v1.5-50_ImageNet_20200110.zip?dl=0) |
| ResNet-v1.5-50 | [77.02%](https://www.dropbox.com/s/2tw1e5w4a48abp7/learning_curve-result-1.svg?dl=0) | 93.24% | + Cosine LR, decoupled WD 4e-5, dropout 0.3 | [#.py](https://www.dropbox.com/s/ru6lmizsw7ck1w4/ResNet-v1.5-50_ImageNet_cos.py?dl=0) | [#.zip](https://www.dropbox.com/s/b1g1wjlmq0ziohj/ResNet-v1.5-50_ImageNet_cos.zip?dl=0) |
| ResNet-v1.5-50 | 77.51% | 93.80% | + Extended crop scale [0.08, 1.0] -> [0.04, 1.96] | #.py | #.zip |
The reported accuracies are for single crop validation images.

### TODO
- Speedup: Training is slower than tf_cnn_benchmark.
- Include the detection task, GAN, etc..
