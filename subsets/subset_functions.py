import numpy as np
import cv2
from skimage import img_as_float32


def random_resized_crop(image, out_size, interpolation=cv2.INTER_LINEAR,
                        random=True, scale=(1.0, 1.0), ratio=(1.0, 1.0)):
    in_size = image.shape
    H = in_size[0]
    W = in_size[1]
    h_ratio = float(H)/out_size[0]
    w_ratio = float(W)/out_size[1]

    if h_ratio < w_ratio:
        re_size = [out_size[0], int(np.round(W/h_ratio))]
    else:
        re_size = [int(np.round(H/w_ratio)), out_size[1]]

    image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
    if random:
        max_size = max(image.shape)
        image = zero_pad(image, (max_size, max_size))

        lower, upper = scale
        a = upper**2 - lower**2
        b = lower**2
        randval = np.random.uniform()
        rand_scale = np.sqrt(a*randval + b)

        lower, upper = ratio
        base = float(upper/lower)
        randval = np.random.uniform()
        rand_ratio = lower*np.power(base, randval)

        rand_x_scale = np.sqrt(rand_scale/rand_ratio)
        rand_y_scale = np.sqrt(rand_scale*rand_ratio)

        size_h = np.around(np.sqrt(H*W)*rand_y_scale).astype(int)
        size_h = min([max_size, size_h])
        size_w = np.around(np.sqrt(H*W)*rand_x_scale).astype(int)
        size_w = min([max_size, size_w])

        offset_h = int(np.random.uniform(0, max_size - size_h))
        offset_w = int(np.random.uniform(0, max_size - size_w))

        image = image[offset_h:offset_h + size_h, offset_w:offset_w + size_w]
        image = cv2.resize(image, dsize=tuple(out_size[1::-1]), interpolation=interpolation)
    else:
        image = crop(image, out_size, random=random)

    return img_as_float32(image)


def resize_with_crop(image, out_size, interpolation=cv2.INTER_LINEAR, random=False):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio < 1.0 or w_ratio < 1.0:
        resize_expand(image, out_size, interpolation=interpolation)
    else:
        image = img_as_float32(image)

    image = crop(image, out_size, random=random)

    return image


def resize_with_pad(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio > 1.0 or w_ratio > 1.0:
        resize_fit(image, out_size, interpolation=interpolation)
    else:
        image = img_as_float32(image)

    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return image


def resize_fit_expand(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio > 1.0 and w_ratio > 1.0:
        image = resize_expand(image, out_size, interpolation=interpolation, random=random)
    elif h_ratio < 1.0 and w_ratio < 1.0:
        image = resize_fit(image, out_size, interpolation=interpolation, random=random)
    else:
        image = img_as_float32(image)

    image = crop(image, out_size, random=random)
    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return image


def resize_fit(image, out_size, interpolation=cv2.INTER_LINEAR, random=False, pad_value=0.0):
    in_size = image.shape
    h_ratio = float(in_size[0]) / out_size[0]
    w_ratio = float(in_size[1]) / out_size[1]

    if h_ratio > w_ratio:
        re_size = [out_size[0], int(np.round(in_size[1] / h_ratio))]
    else:
        re_size = [int(np.round(in_size[0] / w_ratio)), out_size[1]]

    image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return img_as_float32(image)


def resize_expand(image, out_size, interpolation=cv2.INTER_LINEAR, random=False):
    in_size = image.shape
    h_ratio = float(in_size[0])/out_size[0]
    w_ratio = float(in_size[1])/out_size[1]

    if h_ratio < w_ratio:
        re_size = [out_size[0], int(np.round(in_size[1] / h_ratio))]
    else:
        re_size = [int(np.round(in_size[0] / w_ratio)), out_size[1]]

    image = cv2.resize(image, dsize=tuple(re_size[::-1]), interpolation=interpolation)
    image = crop(image, out_size, random=random)

    return img_as_float32(image)


def resize_with_crop_or_pad(image, out_size, random=False, pad_value=0.0, *args, **kwargs):
    image = crop(image, out_size, random=random)
    image = zero_pad(image, out_size, random=random, pad_value=pad_value)

    return img_as_float32(image)


def crop(image, out_size, random=False):
    in_size = image.shape
    out_size = list(out_size)
    h_diff = in_size[0] - out_size[0]
    w_diff = in_size[1] - out_size[1]
    assert h_diff >= 0 or w_diff >= 0, 'At least one side must be longer than or equal to the output size'

    if h_diff > 0:
        if random:
            h_idx = np.random.randint(0, h_diff)
        else:
            h_idx = h_diff//2
        image = image[h_idx:h_idx + out_size[0], :]
    if w_diff > 0:
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
    assert h_diff >= 0 and w_diff >= 0, 'The input size must be smaller than or equal to the output size'

    if random:
        h_idx = np.random.randint(0, h_diff)
        w_idx = np.random.randint(0, w_diff)
    else:
        h_idx = h_diff//2
        w_idx = w_diff//2

    image_out = np.zeros(out_size[:2] + [in_size[-1]], dtype=image.dtype)
    image_out[h_idx:h_idx + in_size[0], w_idx:w_idx + in_size[1]] = image

    return image_out
