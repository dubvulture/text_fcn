#coding=utf-8
from __future__ import division, print_function
from six.moves import xrange

import os

import cv2
import numpy as np

import coco_utils


# Default options for no cropping (therefore batch size = 1)
NO_BATCH_OPT = {
    'batch': 1,
    'size': None
}


class BatchDataset:

    def __init__(self, img_ids, coco_dir, ct, image_options=None):
        """
        Intialize a generic file reader with batching for list of files
        :param img_ids: list of images' coco ids
        :param coco_dir: directory of COCO Dataset
        :param ct: COCO_Text instance
        :param image_options: dictionary of options for cropping images
        Available options:
            batch = # of images for each batch
            size = target (square) size for cropped window
        """
        print("Initializing Batch Dataset Reader...")
        self.img_ids = np.array(img_ids)
        self.coco_dir = coco_dir
        self.ct = ct
        if image_options is None:
            self.image_options = NO_BATCH_OPT
            self._read_images = self._simple_read
        else:
            self.image_options = image_options
            self._read_images = self._batch_read
        print(self.image_options)

        self.images = None
        self.annotations = None
        self.weights = None
        self.batch_offset = 0

    def next_batch(self):
        batch_size = int(self.image_options['batch'])

        if (self.batch_offset + batch_size) > self.img_ids.shape[0]:
            # Epoch finished, shuffle filenames
            np.random.shuffle(self.img_ids)
            # Start next epoch
            self.batch_offset = 0

        batch = slice(self.batch_offset, self.batch_offset+batch_size)
        self.batch_offset += batch_size

        return  self._read_images(batch)

    def get_random_batch(self, batch_size):
        batch = np.random.randint(0, self.img_ids.shape[0], size=[batch_size])
        return self._read_images(batch, batch_size)


    def _batch_read(self, pos):
        """
        Read multiple images/annotations/weights and crop to same size.
        This should be used when random crops are to be used.
        :param pos: slice object
        """
        n = pos.stop - pos.start
        size = self.image_options['size']

        images = np.zeros((n, size, size, 3), dtype=np.float32)
        annotations = np.zeros((n, size, size, 1), dtype=np.uint8)
        weights = np.zeros((n, size, size, 1), dtype=np.float32)
        filenames = np.zeros((n), dtype=object)

        for i, img_id in enumerate(self.img_ids[pos]):
            res = self._gen_images(img_id)
            valid_anns = [
                ann for ann in self.ct.imgToAnns[img_id]
                if self.ct.anns[ann]['legibility'] == 'legible'
            ]
            ann = np.random.choice(valid_anns)
            window = self._get_window(res[1].shape, self.ct.anns[ann])
            image, annotation, weight = self._crop_resize(res, window, size)
            images[i] = image
            annotations[i] = annotation[:,:,None]
            weights[i] = weight[:,:,None]
            filenames = self.ct.imgs[img_id]['file_name'][:-4]

        return [images, annotations, weights, filenames]

    def _simple_read(self, pos):
        """
        Read only 1 image/annotation/weight of any size.
        This should be used only when batch size is equal to 1 and no randomness
        on the dataset is allowed.
        :param pos: slice object of length 1
        """
        assert((pos.stop - pos.start) == 1)
        img_id = self.img_ids[pos][0]
        image, annotation, weight = self._gen_images(img_id)

        # Add batch_dim + convert to floating point in [0,1]
        image = np.expand_dims(image, axis=0)
        # [None,:,:,None] ==> expand_dims(_, axis=[0,3]) // syntax not supported
        annotation = annotation[None,:,:,None]
        weight = weight[None,:,:,None]
        filename = self.ct.imgs[img_id]['file_name'][:-4]

        return [image, annotation, weight, filename]

    def _gen_images(self, img_id):
        """
        :param img_id: image's coco id
        :return: image, its groundtruth w/o illegibles and its weights
        """
        fname = self.ct.imgs[img_id]['file_name']
        image = cv2.imread(
            os.path.join(self.coco_dir, 'images/', fname),
        ).astype(np.float32) / 255.
        annotation = np.zeros(image.shape[:-1], np.uint8)
        weight = np.ones(image.shape[:-1], np.float32)

        for ann in self.ct.imgToAnns[img_id]:
            poly = np.array(self.ct.anns[ann]['polygon'], np.int32).reshape((4,2))

            if self.ct.anns[ann]['legibility'] == 'legible':
                # draw only legible bbox/polygon
                cv2.fillConvexPoly(annotation, poly, 1)
            else:
                # 0 weight if it is illegible
                cv2.fillConvexPoly(weight, poly, 0.0)

        return [image, annotation, weight]

    def _get_window(self, shape, annotation):
        """
        :param shape: image shape (used as boundaries)
        :param annotation:  ct.anns object
        :return: window dict with slices and padding values
        """
        x, y, dx, dy = np.array(annotation['bbox'], np.int32)
        x1, y1, x2, y2 = x, y, x+dx, y+dy
        h, w = shape

        # expand left, right, up, down
        x1 -= np.random.randint(0, x1+1)
        x2 += np.random.randint(0, w-x2+1)
        y1 -= np.random.randint(0, y1+1)
        y2 += np.random.randint(0, h-y2+1)

        ratio = (x2 - x1) / (y2 - y1)

        if ratio > 1:
            # expand ys (window's width)
            diff = (x2 - x1) - (y2 - y1)
            split = np.random.choice(np.arange(0, diff, dtype=np.int32))
            y1 -= split
            y2 += diff-split
        elif ratio < 1:
            # expand xs (window's height)
            diff = (y2 - y1) - (x2 - x1)
            split = np.random.choice(np.arange(0, diff, dtype=np.int32))
            x1 -= split
            x2 += diff-split
        else:
            # already a square
            pass

        assert((x2 - x1) == (y2 - y1))

        return {
            'slice': [
                slice(max(0, y1), y2),
                slice(max(0, x1), x2)
            ],
            'pad': (
                (max(0, -y1), max(0, y2-h)),
                (max(0, -x1), max(0, x2-w))
            )
        }

    def _crop_resize(self, images, window, size):
        """
        :param images: list of (image, annotation, weight)
        :param window: window returned y _get_window()
        :param size: resize size (square)
        :return: cropped images
        """
        pad = [((0,0),), (), ()]
        interp = [cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_NEAREST]
        dsize = (size, size)

        for i in xrange(3):
            images[i] = images[i][window['slice']]
            images[i] = np.pad(images[i], window['pad'] + pad[i], 'constant')
            images[i] = cv2.resize(images[i], dsize, interpolation=interp[i])

        return images
