# MyConvNet
  Deep learning using TensorFlow low-level APIs.

  Build your own convolutional neural networks using TensorFlow.
  
  Supports image classification and semantic segmentation tasks.
  
  DCGAN is now available.
  
  Verified on Windows 10 and Ubuntu 18.04 using PyCharm with Anaconda.
  
  Check out the [instruction](https://www.dropbox.com/s/64wtb6kvn9ms5o3/MyConvNet.pptx?dl=0).
  
  [Getting started with Linux](https://www.dropbox.com/s/uiz95c6vnlsvcp0/TF_Linux_SSH.pptx?dl=0).

## How To Run
- Download all the files.
- Prepare your data using scripts in subsets/.
- Build your own networks by modifying scripts in models/.
- Edit parameters.py to change the dataset, model, directories, etc...
- Run train.py to train the model.
- Run test.py to test the trained model.
- Use inference.py if you have no label for test data.

### Notes
- If you have no NVIDIA GPU, set num_gpus parameter to 0 in order to utilize a CPU for training/inference.
- Our RandomResizedCrop performs padding prior to cropping so that (each side of an image) ≥ √(max_scale·H·W).
  - Use random_resized_crop_nopad in subsets/subset_functions.py for RandomResizedCrop without padding.
- In the segmentation task, pixels with a value of 0 are ignored, so assign 1 to the first class.
- Use Linux for faster training.
- Multi-GPU training is available based on the parameter server strategy.
- NCCL-based distributed training code is curruntly not available (nccl.convnet.py, nccl.optimizer.py).
- Batch statistics of multiple devices are updated successively.
- Check out [REFERENCES.md](https://github.com/dooyounggo/MyConvNet/blob/master/REFERENCES.md) for papers and code references.

### Packages
- Python: 3.7
- tensorflow-gpu: 1.14.0 or 1.15.0 (cudatoolkit: 10.0, cudnn: 7.6.5)
- numpy: 1.17.4
- matplotlib: 3.1.1
- scikit-image: 0.15.0
- scikit-learn: 0.22
- opencv-python: 4.1.2.30

### TODO
- Speedup: Training is slower than tf_cnn_benchmark.
- Include the detection task, GAN, etc..

## Checkpoints
###ImageNet - subtracted by 0.5 and multiplied by 2, ranging in [-1.0, 1.0]
| Model | Top-1 Acc | Top-5 Acc | Train (Test) Image/Input Size | Details | Param | Ckpt |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-v1.5-50 | [76.35%](https://www.dropbox.com/s/4aoscqqovpdaqwr/ResNet-v1.5-50_ImageNet.svg?dl=0) | 92.94% | 224/224 (256/224) | Inception preprocessing (baseline) | [#.py](https://www.dropbox.com/s/lhmnshgfs9jvrfd/imagenet_res50.py?dl=0) | [#.zip](https://www.dropbox.com/s/ruxx6lhmkzmu7u9/ResNet-v1.5-50_ImageNet.zip?dl=0) |
| ResNet-v1.5-50 | [76.50%](https://www.dropbox.com/s/1h8udkqxi97fhg4/learning_curve-result-1.svg?dl=0) | 93.06% | 224/224 (256/224) | + 30 epochs (120 in total) | [#.py](https://www.dropbox.com/s/w197etq5hkl4koy/ResNet-v1.5-50_ImageNet.py?dl=0) | [#.zip](https://www.dropbox.com/s/xl15y6g0n4aaq20/ResNet-v1.5-50_ImageNet_20200110.zip?dl=0) |
| ResNet-v1.5-50 | [77.02%](https://www.dropbox.com/s/2tw1e5w4a48abp7/learning_curve-result-1.svg?dl=0) | 93.24% | 224/224 (256/224) | + Cosine LR, decoupled WD 4e-5, dropout 0.3 | [#.py](https://www.dropbox.com/s/ru6lmizsw7ck1w4/ResNet-v1.5-50_ImageNet_cos.py?dl=0) | [#.zip](https://www.dropbox.com/s/b1g1wjlmq0ziohj/ResNet-v1.5-50_ImageNet_cos.zip?dl=0) |
| ResNet-v1.5-50 | [77.51%](https://www.dropbox.com/s/zd27ccoherakcvz/learning_curve-result-1.svg?dl=0) | 93.80% | 224/224 (256<sup>†</sup>/224) | + Extended crop scale <br> [0.08, 1.0] -> [0.04, 1.96] | [#.py](https://www.dropbox.com/s/qusgwj91mgmml79/ResNet-v1.5-50_ImageNet_es.py?dl=0) | [#.zip](https://www.dropbox.com/s/wqcsb0skk4uvwtn/ResNet-v1.5-50_ImageNet_ES.zip?dl=0) |
| Efficient<br>Net-B0 |  |  |  |  |  |  |

The reported accuracies are single-crop validation scores.

Note that the class numbers are ordered by the [synset IDs](https://www.dropbox.com/s/q8a6epwy2g2phht/synset_words.txt?dl=0) ([train.txt](https://www.dropbox.com/s/bqgzxfv2ys4ej6l/train.txt?dl=0), [val.txt](https://www.dropbox.com/s/wihtxqlotdlmodu/val.txt?dl=0)).

Image size refers to the size after preprocessing. If image and input sizes do not match, cropping or padding is performed.

† Crop method is slightly different, which is center crop of a √(HW) by √(HW) region, zero padding, and resize.
