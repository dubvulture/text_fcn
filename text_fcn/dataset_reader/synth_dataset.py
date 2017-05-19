# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np

from text_fcn import coco_utils
from text_fcn.dataset_reader.dataset_reader import BatchDataset


class SynthDataset(BatchDataset):

        def __init__(self,
                     fnames,
                     st,
                     synth_dir,
                     batch_size,
                     crop_size=0,
                     pre_saved=False):
            """
            :param fnames: 
            :param st: 
            :param synth_dir: 
            :param batch_size: 
            :param crop_size: 
            :param pre_saved: 
            """
            crop_fun = self._crop_resize if crop_size > 0 and not pre_saved else None
            BatchDataset.__init__(self, fnames, batch_size, crop_size, image_op=crop_fun)

            self.st = st
            self.synth_dir = synth_dir
            self.pre_saved = pre_saved

            if self.pre_saved:
                self._get_image = self._load_image
            else:
                self._get_image = self._gen_image

        def _gen_image(self, fname):
            image = cv2.imread(os.path.join(self.synth_dir, 'images/', fname+'.jpg'))
            annotation = np.zeros(image.shape[:-1], dtype=np.uint8)
            weight = np.ones(image.shape[:-1], dtype=np.float32)

            for charBB in self.st[fname]:
                cv2.fillConvexPoly(annotation, charBB.astype(np.int32), 255)

            return [image, annotation, weight]

        def _load_image(self, fname):
            fname += '.png'
            images = cv2.imread(
                os.path.join(self.synth_dir, 'subset_validation/images/', fname))
            annotation = cv2.imread(
                os.path.join(self.synth_dir, 'subset_validation/anns/', fname))
            annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
            weight = np.ones(annotation.shape, np.float32)

            return [images, annotation, weight]

        def _crop_resize(self, image, annotation, weight, name=None):
            assert name is not None
            choice = np.random.randint(0, val[fname].shape[0])
            bbox = np.floor(val[fname][choice])
            # [xi, yi : i=0..4] => x, y, w, h
            bbox = cv2.boundingRect(np.expand_dims(bbox, axis=1))
            bbox = np.int32(bbox)
            window = coco_utils.get_window(annotation.shape, bbox)
            return coco_utils.crop_resize([image, annotation, weight], window, self.crop_size)