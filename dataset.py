"""
Manage datasets using the TensorFlow Data API.
"""

import numpy as np
import tensorflow as tf
import csv
import cv2
import warnings
import subsets.subset_functions as sf


class DataSet(object):
    def __init__(self, image_dirs, label_dirs=None, class_names=None,
                 out_size=None, resize_method=None, resize_randomness=False,
                 **kwargs):
        if label_dirs is None:
            label_dirs = [np.nan for _ in image_dirs]  # Fake labels
        assert len(image_dirs) == len(label_dirs), 'Number of examples mismatch, between images and labels'

        self._image_dirs = image_dirs
        self._label_dirs = label_dirs
        if out_size is None:
            self._image_size = kwargs.get('image_size', (256, 256, 3))
        else:
            self._image_size = out_size

        if resize_method is None:
            self._resize_method = kwargs.get('resize_type', 'resize')
        else:
            self._resize_method = resize_method
        self._resize_randomness = resize_randomness

        if class_names is None:
            self._num_classes = None
        else:
            self._num_classes = len(class_names)
        self._class_names = class_names
        self._num_shards = kwargs.get('num_gpus', 1)
        self._batch_size = kwargs.get('batch_size', 32)
        self._shuffle = kwargs.get('shuffle', True)

        self._parameters = kwargs

        self._image_mean = kwargs.get('image_mean', 0.5)
        self._num_examples = len(image_dirs)
        self._examples_per_class = None
        self._balanced_weights = None

        self._datasets = []
        self._iterators = []
        self._handles = []
        batch_size_per_gpu = self.batch_size//self.num_shards
        with tf.name_scope('dataset/'):
            with tf.device('/cpu:0'):
                for i in range(self.num_shards):
                    dataset = tf.data.Dataset.from_tensor_slices((image_dirs[i::self.num_shards],
                                                                  label_dirs[i::self.num_shards]))
                    if self.shuffle:
                        dataset = dataset.shuffle(buffer_size=min([np.ceil(self.num_examples/self.num_shards),
                                                                   np.ceil(256*batch_size_per_gpu/32)]))
                    dataset = dataset.map(lambda image_dir, label_dir: tuple(tf.py_func(self._load_function,
                                                                                        (image_dir, label_dir),
                                                                                        (tf.float32, tf.float32))),
                                          num_parallel_calls=kwargs.get('num_parallel_calls')//self.num_shards)
                    dataset = dataset.batch(batch_size_per_gpu)
                    dataset = dataset.apply(tf.data.experimental.copy_to_device('/gpu:{}'.format(i)))
                    with tf.device('/gpu:{}'.format(i)):
                        dataset = dataset.prefetch(buffer_size=1)
                        self._datasets.append(dataset)
                        iterator = dataset.make_initializable_iterator()
                        self._iterators.append(iterator)
                        self._handles.append(iterator.string_handle())
        self._init_count = 0

    @property
    def image_dirs(self):
        return self._image_dirs

    @property
    def label_dirs(self):
        return self._label_dirs

    @property
    def image_size(self):
        return self._image_size

    @property
    def num_shards(self):
        return self._num_shards

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def resize_method(self):
        return self._resize_method

    @property
    def resize_randomness(self):
        return self._resize_randomness

    @property
    def image_mean(self):
        return self._image_mean

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_names(self):
        return self._class_names

    @property
    def examples_per_class(self):
        return self._examples_per_class

    @property
    def balanced_weights(self):
        return self._balanced_weights

    @property
    def datasets(self):
        return self._datasets

    @property
    def iterators(self):
        return self._iterators

    @property
    def handles(self):
        return self._handles

    @property
    def init_count(self):
        return self._init_count

    def _load_function(self, image_dir, label_dir):
        if isinstance(image_dir, bytes):
            image_dir = image_dir.decode()
        if isinstance(label_dir, bytes):
            label_dir = label_dir.decode()
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        image = self._resize_function(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        if not isinstance(label_dir, str):
            label = np.array(np.nan, dtype=np.float32)
        else:   # Note that the labels are not one-hot encoded.
            if label_dir.split('.')[-1].lower() == 'csv':   # Classification and detection
                f = open(label_dir, 'r', encoding='utf-8')
                rdr = csv.reader(f)
                line = next(rdr)
                f.close()

                if len(line) == 1:  # Classification
                    label = int(line[0])
                else:
                    label = []      # Detection, TBA
                    for l in line:
                        label.append(int(l))
                label = np.array(label, dtype=np.float32)
            else:   # Segmentation
                label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
                label = label.astype(int) - 1  # Set values corresponding to edge pixels to -1
                label = self._resize_function(label, self.image_size, interpolation=cv2.INTER_NEAREST)
                label = np.round(label[..., 0]*255)

        return image, label

    def _resize_function(self, image, image_size, **kwargs):
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :-1]
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))

        if self.resize_method.lower() == 'resize':
            image = sf.to_float(cv2.resize(image, dsize=tuple(image_size[1::-1]), interpolation=interpolation))
        elif self.resize_method.lower() == 'resize_fit':
            image = sf.resize_fit(image, image_size, interpolation=interpolation, random=self.resize_randomness)
        elif self.resize_method.lower() == 'resize_expand':
            image = sf.resize_expand(image, image_size, interpolation=interpolation, random=self.resize_randomness)
        elif self.resize_method.lower() == 'resize_fit_expand':
            image = sf.resize_fit_expand(image, image_size, interpolation=interpolation, random=self.resize_randomness)
        elif self.resize_method.lower() == 'random_resized_crop' or self.resize_method.lower() == 'random_resize_crop':
            scale_ratio = self._parameters['rand_crop_scale']
            if scale_ratio[1] > 1.0:
                warnings.warn('The maximum scale ratio {} is greater than 1.0.'.format(scale_ratio), UserWarning)
            image = sf.random_resized_crop(image, image_size, interpolation=interpolation, random=self.resize_randomness
                                           , scale=scale_ratio, ratio=self._parameters['rand_crop_ratio'])
        else:
            raise(ValueError, 'Resize type of {} is not supported.'.format(self.resize_method))
        return image

    def data_statistics(self, verbose=True):
        image_mean = 0.0
        examples = np.zeros(self.num_classes, dtype=np.int)
        for idir, ldir in zip(self.image_dirs, self.label_dirs):
            image, label = self._load_function(idir, ldir)

            image_mean += np.mean(image, axis=(0, 1))/self.num_examples

            if len(label.shape) == 0:
                examples[int(label)] += 1
            else:
                for i in range(examples.shape[0]):
                    c = np.equal(label, i).sum()
                    examples[i] += c

        self._image_mean = image_mean
        self._examples_per_class = examples
        self._balanced_weights = self._calculate_balanced_weights()

        if verbose:
            print('Image mean:', image_mean)
            print('Number of examples per class:')
            for i in range(self.num_classes):
                print('{}: {:4d}\t'.format(self.class_names[i], examples[i]), end='')
                if (i + 1) % 5 == 0:
                    print('')
            print('')

    def _calculate_balanced_weights(self):
        w = self.examples_per_class.sum()/self.num_classes/self.examples_per_class
        w[np.where(self.examples_per_class == 0)] = 1.0

        return w

    def initialize(self, sess):
        initializers = []
        for it in self.iterators:
            initializers.append(it.initializer)
        sess.run(initializers)
        self._init_count += 1

    def get_string_handles(self, sess):
        return sess.run(self.handles)
