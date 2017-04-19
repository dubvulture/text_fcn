#coding=utf-8

import os
import numpy as np
import scipy.misc as misc
import coco_utils as utils

"""
Adapted from https://github.com/****/FCN/BatchDatsetReader.py
"""


DEFAULT_OPTIONS = {
    'batch': 2,
    'smin': 128,
    'smax': 512,
    'size': 256
}


class BatchDataset:

    def __init__(self, filenames, directory, image_options=None):
        """
        Intialize a generic file reader with batching for list of files
        :param filenames: list of filenames
        :param directory: directory where to retrive images
        :param image_options: dictionary of options for cropping images
        Available options:
            batch = # of images for each batch
            smin = minimum size for cropping window
            smax = maximum size for cropping window
            size = target (square) size for cropped window
        """
        print("Initializing Batch Dataset Reader...")
        self.dir = directory
        self.image_options = image_options or DEFAULT_OPTIONS
        print(self.image_options)
        self.filenames = np.array(filenames)
        self.images = None
        self.annotations = None
        self.batch_offset = 0

    def _read_images(self, pos, n):
        smin = int(self.image_options['smin'])
        smax = int(self.image_options['smax'])
        size = int(self.image_options['size'])
        
        images = np.zeros((n, size, size, 3))
        annotations = np.zeros((n, size, size, 1))

        for i, filename in enumerate(self.filenames[pos]):
            image = misc.imread(
                    os.path.join(self.dir, 'images/', filename+'.jpg'),
                    mode='RGB')
            mask = misc.imread(
                    os.path.join(self.dir, 'annotations/', filename+'.png'),
                    mode='L')
            mask[mask>0] = 1
            lims = utils.mask_limits(mask)
            # None mask if there is no annotation for given image 
            temp = None if lims==(0,0,0,0) else mask
            ret = utils.randcrop(
                    image.shape, smin, smax, mask=temp, mask_lims=lims)
            images[i] = \
                utils.cropandresize(image, size, *ret)
            annotations[i] = \
                utils.cropandresize(mask, size, *ret, interp='nearest')

        return images, annotations

    def next_batch(self):
        batch_size = int(self.image_options['batch'])

        if self.batch_offset == self.filenames.shape[0]:
            # Epoch finished, shuffle filenames    
            np.random.shuffle(self.filenames)
            # Start next epoch
            self.batch_offset = 0
        elif (self.batch_offset + batch_size) > self.filenames.shape[0]:
            # last step & (batch_size % len) != 0
            batch_size = self.filenames.shape[0] - self.batch_offset

        batch = slice(self.batch_offset, self.batch_offset+batch_size)
        self.batch_offset += batch_size

        self.images, self.annotations = self._read_images(batch, batch_size)
        return self.images, self.annotations, self.filenames[batch]

    def get_random_batch(self, batch_size):
        batch = np.random.randint(0, self.filenames.shape[0], size=[batch_size])
        images, annotations = self._read_images(batch, batch_size)
        return images, annotations, self.filenames[batch]
