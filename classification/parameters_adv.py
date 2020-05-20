"""
Setup various hyperparameters
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from initialization import init_params
from dataset import DataSet
from subsets.cub_200_2011 import read_subset
from models.resnet_v1_5 import ResNet50 as ConvNet
from optimizers import MomentumOptimizer as Optimizer
from evaluators import AccuracyEvaluator as Evaluator
from models.init_from_checkpoint import resnet_v1_50_101 as init_from_checkpoint
import utils


class Parameters(object):
    # FIXME: Directories
    _root_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # FIXME
    _train_dir = os.path.join(_root_dir, 'tfdatasets/cub-200-2011/train')
    _train_sample_size = None  # Data size. None for all files in train_dir
    _val_dir = None  # Can be None
    _val_sample_size = 500
    _test_dir = os.path.join(_root_dir, 'tfdatasets/cub-200-2011/test')
    _test_sample_size = None

    _save_dir = 'D:/trained_models/ResNet-50_CUB-200'
    _transfer_dir = None
    _checkpoint_dir = os.path.join(_root_dir, 'pretrained_models', 'resnet_v1_50', 'resnet_v1_50.ckpt')

    d = dict()
    # FIXME: Image pre-processing hyperparameters
    d['image_size'] = (224, 224, 3)  # Processed image size
    d['image_size_test'] = (256, 256, 3)  # If None, same as 'image_size'
    d['resize_type'] = 'resize_expand'  # Resize types: 'resize', 'resize_expand', 'random_resized_crop', ...
    d['resize_type_test'] = None  # If None, same as 'resize_type'
    d['resize_random'] = True  # Randomness of padding and crop operations
    d['resize_random_test'] = False
    d['resize_interpolation'] = 'bicubic'  # Interpolation methods: 'nearest', 'bilinear', 'bicubic'
    d['rand_resized_crop_scale'] = (0.08, 1.0)  # Scale for 'random_resized_crop' method
    d['rand_resized_crop_ratio'] = (3/4, 4/3)  # Aspect ratio for 'random_resized_crop' method
    d['max_crop_attempts'] = 10  # Maximum number of 'random_resized_crop' attempts to crop a image with a proper size
    d['min_object_size'] = None  # Minimum object area (proportion) of 'random_resized_crop'
    d['padded_resize_scale'] = 2.0  # Scale for 'padded_resize' method. (scale - 1)*num_pixels zeros are padded

    d['input_size'] = (224, 224, 3)  # Network input size after augmentation
    d['image_mean'] = 0.5  # If None, it will be calculated and it may take some time
    d['zero_center'] = True  # Whether to zero-center the images
    d['shuffle'] = True  # Whether to shuffle the data
    d['num_parallel_calls'] = 4  # Number of parallel operations for dataset.map function

    # FIXME: Transfer learning parameters
    d['init_from_public_checkpoint'] = False  # Whether to use pre-trained model in checkpoint_dir

    d['blocks_to_load'] = None  # List of blocks to load variables on. None for all blocks and [None] for logits
    d['load_moving_average'] = False  # Whether to load exponential moving averages of variables onto variables
    d['start_epoch'] = 0  # Start epoch to continue training from
    d['model_to_load'] = 0  # The (n+1)-th best model is loaded. Can be the name of the checkpoint file

    d['blocks_to_train'] = None  # List of blocks to train. None for all blocks and [None] for logits only
    d['update_batch_norm'] = True  # Whether to update batch norm statistics. None to follow blocks_to_train

    # FIXME: Training hyperparameters
    d['half_precision'] = False  # If True, the float16 data type is used
    d['channel_first'] = False  # If True, the "NCHW" format is used instead of "NHWC"
    d['argmax_output'] = False  # If True, the network's output will be argmaxed (output shape=(N, H, W, 1))
    d['cpu_offset'] = 0  # CPU device offset
    d['gpu_offset'] = 0  # GPU device offset
    d['param_device'] = None  # None: param_device is GPU if num_gpus == 1 else CPU. 'cpu0', 'gpu0', ....

    d['num_gpus'] = 1
    d['batch_size'] = 8  # Total batch size (= batch_size_per_gpu*num_gpus)
    d['num_epochs'] = 300
    d['base_learning_rate'] = 0.25  # Learning rate = base_learning_rate*batch_size/256 (linear scaling rule)
    d['momentum'] = 0.9  # Momentum of optimizers
    d['moving_average_decay'] = 0.99  # Decay rate of exponential moving average
    d['batch_norm_decay'] = 0.99  # Decay rate of batch statistics
    d['gradient_threshold'] = None  # Gradient thresholding using global norm. None for no thresholding
    d['loss_weighting'] = None  # None, 'balanced', [class0_weight, class1_weight, ...]. Loss = -w_c*log(y_c)
    d['focal_loss_factor'] = 0.0  # Focal_loss = -log(y_c)*(1 - y_c)^focal_loss_factor
    d['sigmoid_focal_loss_factor'] = 0.0  # SFL = -log(y_c)*(1 - sigmoid(SFL_factor*(y_c - 0.5)) (experimental)
    d['loss_scaling_factor'] = 1  # Loss scaling factor for half precision training

    d['learning_rate_decay_method'] = 'step'  # None, 'step', 'exponential', 'polynomial', 'cosine' (default)
    d['learning_rate_decay_params'] = (0.1, 150, 250)
    d['learning_warmup_epoch'] = 5.0  # Linear warmup epoch

    d['max_to_keep'] = 5  # Maximum number of models to save
    d['score_threshold'] = 0.0  # Model is saved if its score is better by this threshold
    d['validation_frequency'] = None  # Validate every x iterations. None for every epoch
    d['summary_frequency'] = None  # Tensorboard summary every x iterations. None for every epoch
    d['log_trace'] = False  # Whether to log timeline traces in Chrome tracing format.

    # FIXME: Regularization hyperparameters
    d['l1_reg'] = 0.0  # L1 regularization factor
    d['l2_reg'] = 0.001  # L2 regularization factor
    d['base_weight_decay'] = 0e-4  # Decoupled weight decay factor = base_weight_decay*batch_size/256
    d['bias_norm_decay'] = False  # Whether to apply weight decay to biases and variables of normalization layers
    d['weight_decay_scheduling'] = True  # Whether to apply learning rate scheduling to decoupled weight decay
    d['l1_weight_decay'] = False  # If True, l1 weight decay is used: w := w - weight_decay*sign(w) (experimental)
    d['huber_decay_delta'] = None  # Huber loss delta for weight decay with weight standardization (experimental)

    d['label_smoothing'] = 0.0  # Label smoothing factor
    d['dropout_rate'] = 0.2  # Dropout rate
    d['dropout_weights'] = False
    d['dropout_features'] = True
    d['initial_drop_rate'] = 0.0  # Initial drop rate for stochastic depth
    d['final_drop_rate'] = 0.0  # Final drop rate for stochastic depth

    d['feature_reduction_factor'] = 0  # Feature dimensionality reduction factor for small datasets (experimental)

    # FIXME: Data augmentation hyperparameters
    d['augment_train'] = True  # Online augmentation for training data
    d['augment_test'] = False  # Online augmentation for validation and test data

    d['zero_pad_ratio'] = 0.0  # Zero padding ratio = (zero_padded_image_size - nn_input_size)/nn_input_size

    d['rand_blur_stddev'] = 0.0  # Maximum sigma for Gaussian blur
    d['rand_blur_scheduling'] = False  # Augmentation scheduling (experimental)

    d['rand_affine'] = True  # Bool
    d['rand_affine_scheduling'] = False  # Augmentation scheduling (experimental)
    d['rand_scale'] = (1.0, 1.0)  # Minimum and maximum scaling factors (x/y)
    d['rand_ratio'] = (1.0, 1.0)  # Minimum and maximum pixel aspect ratios (x/y)
    d['rand_x_trans'] = 0.0  # Range in proportion (0.2 means +-10%)
    d['rand_y_trans'] = 0.0  # Range in proportion
    d['rand_rotation'] = 0  # Range in degrees
    d['rand_shear'] = 0  # Range in degrees
    d['rand_x_reflect'] = True  # Bool
    d['rand_y_reflect'] = False  # Bool

    d['rand_crop'] = True  # Bool
    d['rand_crop_scheduling'] = False  # Augmentation scheduling (experimental)
    d['rand_crop_scale'] = (0.25, 1.0)  # Scale*input_size patch is cropped from an image
    d['rand_crop_ratio'] = (4/5, 5/4)
    d['extend_bbox_index_range'] = False  # If True, bounding boxes can be snipped, resulting in smaller crop sizes.
    d['rand_interpolation'] = True  # If true, interpolation method is randomly selected from nearest and bilinear

    d['rand_distortion'] = True  # Bool
    d['rand_distortion_scheduling'] = False  # Augmentation scheduling (experimental)
    d['rand_hue'] = 0.2  # Hue range in proportion
    d['rand_saturation'] = (0.8, 1.25)  # Lower and upper bounds of random saturation factors
    d['rand_color_balance'] = (1.0, 1.0)  # Each color channel is multiplied by a random value
    d['rand_equalization'] = 0.0  # Equalization probability
    d['rand_contrast'] = (0.8, 1.25)  # Lower and upper bounds of random contrast factors
    d['rand_brightness'] = 0.2  # Brightness range in proportion
    d['rand_noise_mean'] = 0.0
    d['rand_noise_stddev'] = 0.0
    d['rand_solarization'] = (0.0, 1.0)  # Lower and upper solarization thresholds
    d['rand_posterization'] = (8, 8)  # Lower and upper bounds of posterization bits

    d['cutmix'] = False  # CutMix augmentation
    d['cutmix_scheduling'] = False  # Augmentation scheduling (experimental)

    def __init__(self, parser=None):
        print('Training directory: ', self.train_dir)
        print('Test directory: ', self.test_dir)
        print('Transfer learning directory: ', self.transfer_dir)
        print('Data save directory: ', self.save_dir)
        print()

        init_params(self.d, parser=parser)

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def train_dir(self):
        return self._train_dir

    @property
    def train_sample_size(self):
        return self._train_sample_size

    @property
    def val_dir(self):
        return self._val_dir

    @property
    def val_sample_size(self):
        return self._val_sample_size

    @property
    def test_dir(self):
        return self._test_dir

    @property
    def test_sample_size(self):
        return self._test_sample_size

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def transfer_dir(self):
        return self._transfer_dir

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir
