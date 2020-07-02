"""
Functions for data pre-processing.
Includes various image reading and resizing functions.
"""

import os
import numpy as np
import cv2

INT_TYPES = ('uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64')
FLOAT_TYPES = ('float16', 'float32', 'float64')
IMAGE_FORMATS = ('bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'jp2', 'png',
                 'webp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif')


def read_subset_cls(subset_dir, shuffle=False, sample_size=None, image_dir=None, label_dir=None):
    filenames = os.listdir(subset_dir)
    filenames.sort()
    image_paths = []
    label_paths = []
    for fname in filenames:
        ext = fname.split('.')[-1].lower()
        full_filename = os.path.join(subset_dir, fname)
        if ext == 'csv':
            label_paths.append(full_filename)
        elif ext in IMAGE_FORMATS:
            image_paths.append(full_filename)

    if image_dir is not None:
        full_filenames = []
        recursive_search([image_dir], full_filenames)
        image_paths = []
        for fname in full_filenames:
            ext = fname.split('.')[-1].lower()
            if ext in IMAGE_FORMATS:
                image_paths.append(fname)

    if label_dir is not None:
        full_filenames = []
        recursive_search([label_dir], full_filenames)
        label_paths = []
        for fname in full_filenames:
            ext = fname.split('.')[-1].lower()
            if ext in ('csv', 'txt'):
                label_paths.append(fname)

    if len(label_paths) == 0:
        label_paths = None
    else:
        assert len(image_paths) == len(label_paths), \
            'Number of examples mismatch: {} images vs. {} labels'.format(len(image_paths), len(label_paths))

    set_size = len(image_paths)
    if sample_size is not None and sample_size < set_size:
        if shuffle:
            idx = np.random.choice(np.arange(set_size), size=sample_size, replace=False).astype(int)
            image_paths = list(np.array(image_paths)[idx])
            if label_paths is not None:
                label_paths = list(np.array(label_paths)[idx])
        else:
            image_paths = image_paths[:sample_size]
            if label_paths is not None:
                label_paths = label_paths[:sample_size]
    else:
        if shuffle:
            idx = np.arange(set_size)
            np.random.shuffle(idx)
            image_paths = list(np.array(image_paths)[idx])
            if label_paths is not None:
                label_paths = list(np.array(label_paths)[idx])

    return image_paths, label_paths


def read_subset_seg(subset_dir, shuffle=False, sample_size=None, image_dir=None, label_dir=None):
    filenames = os.listdir(subset_dir)
    filenames.sort()
    image_paths = []
    label_paths = []
    for fname in filenames:
        ext = fname.split('.')[-1].lower()
        full_filename = os.path.join(subset_dir, fname)
        if ext == 'png':
            label_paths.append(full_filename)
        elif ext in ('jpeg', 'jpg', 'bmp'):
            image_paths.append(full_filename)

    if image_dir is not None:
        full_filenames = []
        recursive_search([image_dir], full_filenames)
        image_paths = []
        for fname in full_filenames:
            ext = fname.split('.')[-1].lower()
            if ext in IMAGE_FORMATS:
                image_paths.append(fname)

    if label_dir is not None:
        full_filenames = []
        recursive_search([label_dir], full_filenames)
        label_paths = []
        for fname in full_filenames:
            ext = fname.split('.')[-1].lower()
            if ext in IMAGE_FORMATS:
                label_paths.append(fname)

    if len(label_paths) == 0:
        label_paths = None
    else:
        assert len(image_paths) == len(label_paths), \
            'Number of examples mismatch: {} images vs. {} labels'.format(len(image_paths), len(label_paths))

    set_size = len(image_paths)
    if sample_size is not None and sample_size < set_size:
        if shuffle:
            idx = np.random.choice(np.arange(set_size), size=sample_size, replace=False).astype(int)
            image_paths = list(np.array(image_paths)[idx])
            if label_paths is not None:
                label_paths = list(np.array(label_paths)[idx])
        else:
            image_paths = image_paths[:sample_size]
            if label_paths is not None:
                label_paths = label_paths[:sample_size]
    else:
        if shuffle:
            idx = np.arange(set_size)
            np.random.shuffle(idx)
            image_paths = list(np.array(image_paths)[idx])
            if label_paths is not None:
                label_paths = list(np.array(label_paths)[idx])

    return image_paths, label_paths


def recursive_search(dirs, file_list):
    for path in dirs:
        if os.path.isdir(path):
            sub_dirs = os.listdir(path)
            if sub_dirs:
                sub_dirs.sort()
                sub_dirs = [os.path.join(path, sd) for sd in sub_dirs]
                recursive_search(sub_dirs, file_list)
        else:
            file_list.append(path)


def random_resized_crop(image, out_size, interpolation=cv2.INTER_LINEAR, random=True, scale=(1.0, 1.0),
                        ratio=(1.0, 1.0), max_attempts=10, min_object_size=None, padding=True, pad_value=0.0):

    out_ratio = out_size[1]/out_size[0]
    in_size = image.shape
    h = in_size[0]
    w = in_size[1]
    max_scale = np.sqrt(scale[1])

    size_h = np.sqrt(h*w/out_ratio)
    size_w = np.sqrt(h*w*out_ratio)
    if padding:
        padded_h = max(h, np.ceil(size_h*max_scale).astype(int))
        padded_w = max(w, np.ceil(size_w*max_scale).astype(int))
        image = zero_pad(image, [padded_h, padded_w], pad_value=pad_value)
        offset_x = (padded_w - w)//2
        offset_y = (padded_h - h)//2
    else:
        padded_h = h
        padded_w = w
        offset_x = 0
        offset_y = 0

    scale_augment = scale[0] != 1.0 or scale[1] != 1.0
    ratio_augment = ratio[0] != 1.0 or ratio[1] != 1.0
    augment = scale_augment or ratio_augment
    if random and augment:
        lower, upper = scale
        rand_scale = np.random.uniform(lower, upper)

        lower, upper = ratio
        base = upper/lower
        randval = np.random.uniform()
        rand_ratio = lower*np.power(base, randval)

        rand_x_scale = np.sqrt(rand_scale/rand_ratio)
        rand_y_scale = np.sqrt(rand_scale*rand_ratio)

        size_h = np.around(size_h*rand_y_scale).astype(int)
        size_w = np.around(size_w*rand_x_scale).astype(int)

        success = False
        if min_object_size is None:
            min_object_size = scale[0]
        min_object_area = min_object_size*h*w
        i = 0
        while i < max_attempts and rand_scale > min_object_size and not success:
            x = np.random.randint(w) + offset_x
            y = np.random.randint(h) + offset_y
            x_min = x - size_w//2
            x_max = np.minimum(x_min + size_w, padded_w)
            x_min = np.maximum(0, x_min)
            y_min = y - size_h//2
            y_max = np.minimum(y_min + size_h, padded_h)
            y_min = np.maximum(0, y_min)

            crop_h = y_max - y_min
            crop_w = x_max - x_min
            crop_area = crop_h*crop_w
            crop_ratio = crop_h/crop_w*out_ratio
            if crop_area >= min_object_area and ratio[0] <= crop_ratio <= ratio[1]:
                image = image[y_min:y_max, x_min:x_max]
                success = True
                break
            else:
                i += 1
        if not success:
            image = crop(image, [size_h, size_w], random=random)
    else:
        size_h = np.around(size_h).astype(int)
        size_w = np.around(size_w).astype(int)
        image = crop(image, [size_h, size_w], random=random)

    image = cv2.resize(image, dsize=tuple(out_size[1::-1]), interpolation=interpolation)

    return to_float(image)


def random_resized_crop_nopad(image, out_size, interpolation=cv2.INTER_LINEAR, random=True,
                              scale=(1.0, 1.0), ratio=(1.0, 1.0), max_attempts=10, min_object_size=None):

    out_ratio = out_size[1]/out_size[0]
    in_size = image.shape
    h = in_size[0]
    w = in_size[1]

    size_h = np.sqrt(h*w/out_ratio)
    size_w = np.sqrt(h*w*out_ratio)

    scale_augment = scale[0] != 1.0 or scale[1] != 1.0
    ratio_augment = ratio[0] != 1.0 or ratio[1] != 1.0
    augment = scale_augment or ratio_augment
    if random and augment:
        lower, upper = scale
        rand_scale = np.random.uniform(lower, upper)

        lower, upper = ratio
        base = upper/lower
        randval = np.random.uniform()
        rand_ratio = lower*np.power(base, randval)

        rand_x_scale = np.sqrt(rand_scale/rand_ratio)
        rand_y_scale = np.sqrt(rand_scale*rand_ratio)

        size_h = np.around(size_h*rand_y_scale).astype(int)
        size_w = np.around(size_w*rand_x_scale).astype(int)

        success = False
        if min_object_size is None:
            min_object_size = scale[0]
        min_object_area = min_object_size*h*w
        i = 0
        while i < max_attempts and rand_scale > min_object_size and not success:
            x = np.random.randint(w)
            y = np.random.randint(h)
            x_min = x - size_w//2
            x_max = np.minimum(x_min + size_w, w)
            x_min = np.maximum(0, x_min)
            y_min = y - size_h//2
            y_max = np.minimum(y_min + size_h, h)
            y_min = np.maximum(0, y_min)

            crop_h = y_max - y_min
            crop_w = x_max - x_min
            crop_area = crop_h*crop_w
            crop_ratio = crop_h/crop_w*out_ratio
            if crop_area >= min_object_area and ratio[0] <= crop_ratio <= ratio[1]:
                image = image[y_min:y_max, x_min:x_max]
                success = True
                break
            else:
                i += 1
        if not success:
            image = crop(image, [size_h, size_w], random=random)
    else:
        size_h = np.around(size_h).astype(int)
        size_w = np.around(size_w).astype(int)
        image = crop(image, [size_h, size_w], random=random)

    image = cv2.resize(image, dsize=tuple(out_size[1::-1]), interpolation=interpolation)

    return to_float(image)


def padded_resize(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, scale=2.0, pad_value=0.0):
    scaled_out_size = [np.around(out_size[0]*np.sqrt(scale)).astype(int),
                       np.around(out_size[1]*np.sqrt(scale)).astype(int)]
    in_size = image.shape
    in_size_ratio = in_size[1]/in_size[0]

    size_h = np.sqrt(out_size[0]*out_size[1]/in_size_ratio).astype(int)
    size_w = np.sqrt(out_size[0]*out_size[1]*in_size_ratio).astype(int)
    image = cv2.resize(image, dsize=(size_w, size_h), interpolation=interpolation)
    if size_h > scaled_out_size[0] or size_w > scaled_out_size[1]:
        image = crop(image, scaled_out_size, random=random)
    image = zero_pad(image, scaled_out_size, random=random, pad_value=pad_value)

    return to_float(image)


def resize_with_crop(image, out_size, interpolation=cv2.INTER_LINEAR, random=False):
    in_size = image.shape
    h_ratio = in_size[0]/out_size[0]
    w_ratio = in_size[1]/out_size[1]

    if h_ratio < 1.0 or w_ratio < 1.0:
        resize_expand(image, out_size, interpolation=interpolation)
    else:
        image = to_float(image)

    image = crop(image, out_size, random=random)

    return image


def resize_with_pad(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = in_size[0]/out_size[0]
    w_ratio = in_size[1]/out_size[1]

    if h_ratio > 1.0 or w_ratio > 1.0:
        resize_fit(image, out_size, interpolation=interpolation)
    else:
        image = to_float(image)

    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return image


def resize_fit_expand(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = in_size[0]/out_size[0]
    w_ratio = in_size[1]/out_size[1]

    if h_ratio == 1.0 and w_ratio == 1.0:
        return to_float(image)
    else:
        if h_ratio > 1.0 and w_ratio > 1.0:
            image = resize_expand(image, out_size, interpolation=interpolation, random=random)
        elif h_ratio < 1.0 and w_ratio < 1.0:
            image = resize_fit(image, out_size, interpolation=interpolation, random=random)
        else:
            image = to_float(image)

        image = crop(image, out_size, random=random)
        image = zero_pad(image, out_size, random=random, pad_value=pad_value)

        return image


def resize_fit(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = in_size[0]/out_size[0]
    w_ratio = in_size[1]/out_size[1]

    if h_ratio == 1.0 and w_ratio == 1.0:
        return to_float(image)
    else:
        if h_ratio > w_ratio:
            re_size = [out_size[0], int(np.round(in_size[1]/h_ratio))]
        else:
            re_size = [int(np.round(in_size[0]/w_ratio)), out_size[1]]

        image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
        image = zero_pad(image, out_size, random=random, pad_value=pad_value)

        return to_float(image)


def resize_expand(image, out_size, interpolation=cv2.INTER_LINEAR, random=False):
    in_size = image.shape
    h_ratio = in_size[0]/out_size[0]
    w_ratio = in_size[1]/out_size[1]

    if h_ratio == 1.0 and w_ratio == 1.0:
        return to_float(image)
    else:
        if h_ratio < w_ratio:
            re_size = [out_size[0], int(np.round(in_size[1]/h_ratio))]
        else:
            re_size = [int(np.round(in_size[0]/w_ratio)), out_size[1]]

        image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
        image = crop(image, out_size, random=random)

        return to_float(image)


def resize_with_crop_or_pad(image, out_size, random=False, pad_value=0.0, *args, **kwargs):
    if image.shape[0] > out_size[0] or image.shape[1] > out_size[1]:
        image = crop(image, out_size, random=random)
    if image.shape[0] < out_size[0] or image.shape[1] < out_size[1]:
        image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return to_float(image)


def crop(image, out_size, random=False):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = in_size[0] - out_size[0]
    w_diff = in_size[1] - out_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be longer than or equal to the output size'

    if h_diff > 0 and w_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
            w_idx = np.random.randint(0, w_diff)
        else:
            h_idx = h_diff//2
            w_idx = w_diff//2
        image = image[h_idx:h_idx + out_size[0], w_idx:w_idx + out_size[1]]
    elif h_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
        else:
            h_idx = h_diff//2
        image = image[h_idx:h_idx + out_size[0], :]
    elif w_diff > 0:
        if random:
            w_idx = np.random.randint(0, w_diff)
        else:
            w_idx = w_diff//2
        image = image[:, w_idx:w_idx + out_size[1]]

    return image


def zero_pad(image, out_size, random=False, pad_value=0.0):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = out_size[0] - in_size[0]
    w_diff = out_size[1] - in_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be shorter than or equal to the output size'

    out_size_max = [max(out_size[0], in_size[0]), max(out_size[1], in_size[1])]
    image_out = np.zeros(out_size_max + [in_size[-1]], dtype=image.dtype)
    if pad_value != 0.0:
        image_out += np.array(pad_value, dtype=image.dtype)

    if h_diff > 0 and w_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
            w_idx = np.random.randint(0, w_diff)
        else:
            h_idx = h_diff//2
            w_idx = w_diff//2
        image_out[h_idx:h_idx + in_size[0], w_idx:w_idx + in_size[1]] = image
    elif h_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
        else:
            h_idx = h_diff//2
        image_out[h_idx:h_idx + in_size[0], :] = image
    elif w_diff > 0:
        if random:
            w_idx = np.random.randint(0, w_diff)
        else:
            w_idx = w_diff//2
        image_out[:, w_idx:w_idx + in_size[1]] = image
    else:
        image_out = image

    return image_out


def to_float(image):
    if image.dtype in INT_TYPES:
        image = image.astype(np.float32)/255.0
    else:
        return image.astype(np.float32)
    return image


def to_int(image):
    if image.dtype in FLOAT_TYPES:
        image = (image*255).astype(np.int32)
    else:
        image = image.astype(np.int32)
    return image
