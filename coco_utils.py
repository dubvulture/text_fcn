#coding=utf-8

import os
import sys
import math
import random
import tarfile
import zipfile
import numpy as np
import scipy.misc as misc
from six.moves import urllib

import coco_text as cot


URL = 'https://s3.amazonaws.com/cocotext/COCO_Text.zip'


def read_dataset(directory):
	"""
	Returns train and validation dataset filenames list (no extension)
	:param directory: folder where to find needed files
	"""
	maybe_download_and_extract(directory, URL, is_zipfile=True)
	ct = cot.COCO_Text(os.path.join(directory, 'COCO_Text.json'))
	
	train = [ct.imgs[i]['file_name'][:-4] for i in ct.train if ct.imgToAnns[i] != []]
	val = [ct.imgs[i]['file_name'][:-4] for i in ct.val if ct.imgToAnns[i] != []]
    test = [ct.imgs[i]['file_name'][:-4] for i in ct.test]

	return train, val, test

def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

def to_gt(img):
    """
    :param img: B/W image, values in {0, 255}
    :return:    image in {0, 1}
    """
    img[img > 0] = 1
    return img.astype(np.int32)

def to_mask(res):
    """
    :param res: resulting image from to_gt, values in {0, 1}
    :return:    image in {0, 255}
    """
    res[res > 0] = 255
    return res.astype(np.uint8)

def randcrop(shape, size_min, size_max, mask=None, mask_lims=None):
    height,width = shape[0:2]

    if mask is not None:
        # Prepare checks on the minimum number of pixels in a mask
        # mask_side_original / resize_coefficient >= mask_side_minimum
        # =>
        # mask_side_actual * (size_final / size_initial) >= mask_side_minimum
        # =>
        # (mask_side_actual * size_final) / mask_side_minimum >= size_initial

        # The check is applied by bounding the crop size to a maximum value that would
        # guarantee a minum number of white pixels in the map, following the formula above
        mask_min = 25
        size_final = size_min
        size_max = max(size_min,min(size_max, math.floor(math.sqrt(mask.sum()/255)*size_final/math.sqrt(mask_min))))

        # Also bound the possible left/top to the bounding box of the provided mask
        rmin, rmax, cmin, cmax = mask_lims or mask_limits(mask)

    cropped_mask = np.zeros(1)
    while(cropped_mask.sum() == 0):
        size = random.randint(size_min, size_max)
        if width>=size:
            if mask is None:
                left = random.randint(0, width-size)
            else:
                left = random.randint(max(0,cmin-size+2), min(width-size,cmax-2))
        else:
            # If the crop size is larger than one of the image dimensions,
            # the offset (left or top) will be negative, indicating that padding will be used
            left = -random.randint(0, size-width)
        if height>=size:
            if mask is None:
                top = random.randint(0, height-size)
            else:
                top = random.randint(max(0,rmin-size+2), min(height-size,rmax-2))
        else:
            top = -random.randint(0, size-height)
        # Crop the mask to check that it is not empty
        if mask is not None and mask.sum() > 0:
            cropped_mask = mask[max(0,top):min(height,top+size), max(0,left):min(width,left+size)]
        else:
            cropped_mask[0] = 1

    return size, left, top

def cropandresize(image, res_size, crop_size, left, top, interp='bicubic', mode=None):
    # Pad
    pad_top = max(0, -top)
    pad_bottom = max(0, crop_size - (pad_top + image.shape[0]))
    pad_left = max(0, -left)
    pad_right = max(0, crop_size - (pad_left + image.shape[1]))
    top = max(0,top)
    left = max(0,left)
    if len(image.shape)==3:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), 'constant')
    else:
        padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')

    # Crop
    cropped = padded[top:top+crop_size, left:left+crop_size]
    # Resize
    resized = misc.imresize(cropped, [res_size, res_size], interp=interp, mode=mode)

    # (x,y) => (x,y,1)
    if len(image.shape)==2:
        resized = np.expand_dims(resized, axis=3)
        resized = to_gt(resized)

    return resized

def mask_limits(mask):
    if type(mask) is np.ndarray and mask.sum() > 0:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (rmin, rmax, cmin, cmax)
    else:
        return (0,0,0,0)
