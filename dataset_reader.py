#coding=utf-8

import os
import numpy as np
import scipy.misc as misc
import coco_utils

"""
Adapted from https://github.com/****/FCN/BatchDatsetReader.py
"""


# Default options for no cropping (therefore batch size = 1)
NO_BATCH_OPT = {
    'batch': 1,
    'smin': None,
    'smax': None,
    'size': None
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
        self.filenames = np.array(filenames)
        self.images = None
        self.annotations = None
        self.batch_offset = 0
        if image_options is None:
            self.image_options = NO_BATCH_OPT
            self._read_images = self._simple_read
        else:
            self.image_options = image_options
            self._read_images = self._batch_read
        print(self.image_options)


    def _batch_read(self, pos):
        """
        Read multiple images/annotations and crop to same size.
        This should be used when random crops are to be used.
        :param pos: slice object
        """
        n = pos.stop - pos.start
        smin = self.image_options['smin']
        smax = self.image_options['smax']
        size = self.image_options['size']

        images = np.zeros((n, size, size, 3), dtype=np.float32)
        annotations = np.zeros((n, size, size, 1), dtype=np.float32)

        for i, filename in enumerate(self.filenames[pos]):
            image = misc.imread(
                os.path.join(self.dir, 'images/', filename+'.jpg'),
                mode='RGB')
            ann = misc.imread(
                os.path.join(self.dir, 'annotations/', filename+'.png'),
                mode='L')
            ann[ann>0] = 1

            lims = coco_utils.mask_limits(ann)
            # None mask if there is no annotation for given image 
            mask = None if lims==(0,0,0,0) else ann
            ret = coco_utils.randcrop(
                    image.shape, smin, smax, mask=mask, mask_lims=lims)
            images[i] = \
                coco_utils.cropandresize(image, size, *ret)
            annotations[i] = \
                coco_utils.cropandresize(ann, size, *ret, interp='nearest')


        return images, annotations

    def _simple_read(self, pos):
        """
        Read only 1 image/annotation of any size.
        This should be used only when batch size is equal to 1 and no randomness
        on the dataset is allowed.
        :param pos: slice object of size 1
        """
        assert((pos.stop - pos.start) == 1)

        filename = self.filenames[pos][0]
        image = misc.imread(
            os.path.join(self.dir, 'images/', filename+'.jpg'),
            mode='RGB')
        ann = misc.imread(
            os.path.join(self.dir, 'annotations/', filename+'.png'),
            mode='L')
        ann[ann>0] = 1

        # Add batch_dim + convert to floating point in [0,1]
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255.
        # [None,:,:,None] ==> expand_dims(_, axis=[0,3]) // syntax not supported
        ann = coco_utils.to_mask(ann)[None,:,:,None]

        return image, ann

    def next_batch(self):
        batch_size = int(self.image_options['batch'])

        if (self.batch_offset + batch_size) > self.filenames.shape[0]:
            # Epoch finished, shuffle filenames    
            np.random.shuffle(self.filenames)
            # Start next epoch
            self.batch_offset = 0

        batch = slice(self.batch_offset, self.batch_offset+batch_size)
        self.batch_offset += batch_size

        self.images, self.annotations = self._read_images(batch)
        return self.images, self.annotations, self.filenames[batch]

    def get_random_batch(self, batch_size):
        batch = np.random.randint(0, self.filenames.shape[0], size=[batch_size])
        images, annotations = self._read_images(batch, batch_size)
        return images, annotations, self.filenames[batch]
