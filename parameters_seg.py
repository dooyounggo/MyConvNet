"""
Setup various (hyper)parameters
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import DataSet
import subsets.cityscapes as subset
from models.deeplabv3plus import DeepLabV3PlusResNet as ConvNet
from optimizers import MomentumOptimizer as Optimizer
from evaluators import MeanIoUBEvaluator as Evaluator
from models.init_from_checkpoint import resnet_v2_50_101 as init_from_pretrained_model
import utils


class Parameters(object):
    # FIXME: Directories
    _root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))     # Parent directory  # FIXME
    _train_dir = os.path.join(_root_dir, 'tfdatasets/oxford-iiit-pet-dataset-sualab/train')
    _train_sample_size = None   # Data size. None for all files in train_dir
    _val_dir = None  # Can be None
    _val_sample_size = None
    _test_dir = os.path.join(_root_dir, 'tfdatasets/oxford-iiit-pet-dataset-sualab/test')
    _test_sample_size = None
    _save_dir = 'D:/trained_models/gcn_iiitdogcat-384'

    _transfer_dir = None
    _pretrained_dir = os.path.join(_root_dir, 'pretrained_models', 'resnet_v2_50_2017_04_14', 'resnet_v2_50.ckpt')

    d = dict()
    # FIXME: Image pre-processing (hyper)parameters
    d['image_size'] = (576, 576, 3)  # Processed image size
    d['image_size_test'] = None  # If None, same as 'image_size'
    d['resize_type'] = 'resize_fit'  # Resize types: 'resize', 'resize_expand', 'random_resized_crop', ...
    d['resize_type_test'] = None  # If None, same as 'resize_type'
    d['resize_random'] = False  # Randomness of padding and crop operations
    d['resize_random_test'] = False
    d['resize_interpolation'] = 'bilinear'  # Interpolation methods: 'nearest', 'bilinear', 'bicubic'
    d['rand_resized_crop_scale'] = (0.08, 1.0)  # Scale for 'random_resized_crop' method
    d['rand_resized_crop_ratio'] = (3/4, 4/3)  # Aspect ratio for 'random_resized_crop' method
    d['padded_resize_scale'] = 2.0  # Scale for 'padded_resize' method. (scale - 1)*num_pixels zeros are padded

    d['input_size'] = (480, 480, 3)  # Network input size after augmentation
    d['image_mean'] = 0.5  # If None, it will be calculated and it may take some time
    d['zero_center'] = True  # Whether to zero-center the images
    d['shuffle'] = True  # Whether to shuffle the data
    d['num_parallel_calls'] = 4  # Number of parallel operations for dataset.map function

    # FIXME: Transfer learning parameters
    d['init_from_pretrained_model'] = False  # Whether to use pre-trained model in _pretrained_dir
    d['blocks_to_load'] = None  # Blocks to load variables on. None for all blocks
    d['load_logits'] = False  # Whether to load variables related to logits
    d['load_moving_average'] = True  # Whether to load exponential moving averages of variables onto variables
    d['start_epoch'] = 0  # Start epoch to continue training from
    d['model_to_load'] = 0  # The (n+1)-th best model is loaded for transfer learning and test.

    d['blocks_to_train'] = None  # Blocks to train. None for all blocks and [None] for logits only
    d['update_batch_norm'] = False  # Whether to update batch norm gamma and beta. None to follow blocks_to_train

    # FIXME: Training hyperparameters
    d['half_precision'] = False  # Try half-precision if your GPU supports it
    d['channel_first'] = True  # If true, NCHW format is used instead of NHWC
    d['num_gpus'] = 1
    d['cpu_offset'] = 0  # Offset for selecting CPU
    d['gpu_offset'] = 0  # Offset for selecting GPU numbers
    d['batch_size'] = 4
    d['num_epochs'] = 30
    d['validation_frequency'] = None  # Validate every x iterations. None for every epoch
    d['summary_frequency'] = None  # Tensorboard summary every x iterations. None for every epoch

    d['base_learning_rate'] = 0.05  # Learning rate = base_learning_rate*batch_size/256
    d['momentum'] = 0.9
    d['moving_average_decay'] = 0.9999
    d['batch_norm_decay'] = 0.997
    d['gradient_threshold'] = 5.0  # Gradient thresholding using global norm. None for no thresholding
    d['loss_weighting'] = None  # None, 'balanced', [class0_weight, class1_weight, ...]. Loss = -w_c*log(y_c)
    d['focal_loss_factor'] = 0.0  # Focal_loss = -log(y_c)*(1 - y_c)^focal_loss_factor
    d['sigmoid_focal_loss_factor'] = 0.0  # SFL = -log(y_c)*(1 - sigmoid(SFL_factor*(y_c - 0.5)) (experimental)
    d['loss_scaling_factor'] = 1  # Loss scaling factor for half precision training

    d['learning_rate_decay_method'] = 'cosine'  # 'step', 'exponential', 'cosine' (default)
    d['learning_rate_decay_params'] = (0.94, 2)
    d['learning_warmup_epoch'] = 5.0  # Linear warmup epoch

    d['max_to_keep'] = 5  # Maximum number of models to save
    d['score_threshold'] = 0.0  # A model is saved if its score is better by this threshold

    # FIXME: Regularization hyperparameters
    d['l1_reg'] = 0.0  # L1 regularization factor
    d['l2_reg'] = 0.0  # L2 regularization factor
    d['base_weight_decay'] = 0.00001  # Decoupled weight decay factor = base_weight_decay*batch_size/256
    d['huber_decay_delta'] = None  # Huber loss delta for weight decay with weight standardization (experimental)
    d['label_smoothing'] = 0.0  # Label smoothing factor
    d['dropout_rate'] = 0.0  # Dropout rate
    d['dropout_weights'] = False
    d['dropout_features'] = False
    d['initial_drop_rate'] = 0.0  # Initial drop rate for stochastic depth
    d['final_drop_rate'] = 0.0  # Final drop rate for stochastic depth

    d['feature_reduction_factor'] = 0  # Feature dimensionality reduction factor for small datasets

    # FIXME: Data augmentation (hyper)parameters
    d['augment_train'] = True  # Online augmentation for training data
    d['augment_test'] = False  # Online augmentation for validation or test data

    d['zero_pad_ratio'] = 0.0  # Zero padding ratio = (zero_padded_image_size - nn_input_size)/nn_input_size

    d['rand_blur_stddev'] = 0.0  # Maximum sigma for Gaussian blur

    d['rand_affine'] = True  # Bool
    d['rand_scale'] = (1.0, 1.0)  # Minimum and maximum scaling factors (x/y)
    d['rand_ratio'] = (1.0, 1.0)  # Minimum and maximum pixel aspect ratios (x/y)
    d['rand_x_trans'] = 0.0  # Range in proportion (0.2 means +-10%)
    d['rand_y_trans'] = 0.0  # Range in proportion
    d['rand_rotation'] = 90  # Range in degrees
    d['rand_shear'] = 30  # Range in degrees
    d['rand_x_reflect'] = True  # Bool
    d['rand_y_reflect'] = False  # Bool

    d['rand_crop'] = True  # Bool
    d['rand_crop_scale'] = (0.3**2, 1.2**2)  # Scale*input_size patch crop from an image
    d['rand_crop_ratio'] = (3/4, 4/3)
    d['rand_interpolation'] = True  # If true, interpolation method is randomly selected between nearest and bilinear

    d['rand_distortion'] = True  # Bool
    d['rand_hue'] = 0.2  # Hue range in proportion
    d['rand_saturation'] = (0.625, 1.5)  # Lower and upper bounds of random saturation factors
    d['rand_color_balance'] = (1.0, 1.0)  # Each color channel is multiplied by a random value
    d['rand_equalization'] = 0.0  # Equalization probability
    d['rand_contrast'] = (0.625, 1.5)  # Lower and upper bounds of random contrast factors
    d['rand_brightness'] = 0.2  # Brightness range in proportion
    d['rand_noise_mean'] = 0.0
    d['rand_noise_stddev'] = 0.0
    d['rand_solarization'] = (0.0, 1.0)  # Lower and upper solarization thresholds
    d['rand_posterization'] = (5, 8)  # Lower and upper bounds of posterization bits

    d['cutmix'] = False  # CutMix augmentation

    def __init__(self):
        print('Training directory: ', self.train_dir)
        print('Test directory: ', self.test_dir)
        print('Transfer learning directory: ', self.transfer_dir)
        print('Data save directory: ', self.save_dir)
        print('')

        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(self.d['num_gpus'])
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        # os.environ["OMP_NUM_THREADS"] = str(self.d['num_parallel_calls'])
        # os.environ["KMP_BLOCKTIME"] = '0'
        # os.environ["KMP_SETTINGS"] = '1'
        # os.environ["KMP_AFFINITY"] = 'granularity=fine,verbose,compact,1,0'

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
    def pretrained_dir(self):
        return self._pretrained_dir
