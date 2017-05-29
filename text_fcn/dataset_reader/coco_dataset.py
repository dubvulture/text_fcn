# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np

from text_fcn import coco_utils
from text_fcn.dataset_reader.dataset_reader import BatchDataset


class CocoDataset(BatchDataset):

    def __init__(self,
                 coco_ids,
                 ct,
                 coco_dir,
                 batch_size,
                 crop_size=0,
                 pre_saved=False):
        """
        :param coco_ids:
        :param batch_size:
        :param ct: COCO_Text object instance
        :param coco_dir: directory to coco dataset
        :param crop_size: window cropping size on each image (0 if none)
        :param pre_saved: whether to read images from storage or generate them on the go
        """
        # crop only when crop_size if given AND images are not loaded from disk
        crop_fun = self._crop_resize if crop_size > 0 and not pre_saved else None
        BatchDataset.__init__(self, coco_ids, batch_size, crop_size, image_op=crop_fun)

        self.ct = ct
        self.coco_dir = coco_dir
        self.pre_saved = pre_saved

        if self.pre_saved:
            self._get_image = self._load_image
        else:
            self._get_image = self._gen_image

    def _gen_image(self, coco_id):
        """
        Generate images using self.ct data
        :param coco_id: image's coco id
        :return: image, its groundtruth w/o illegibles and its weights
        """
        fname = self.ct.imgs[coco_id]['file_name']
        image = cv2.imread(
            os.path.join(self.coco_dir, 'images/', fname)
        )
        annotation = np.zeros(image.shape[:-1], dtype=np.uint8)
        weight = np.ones(annotation.shape, np.float32)

        for ann in self.ct.imgToAnns[coco_id]:
            poly = np.array(self.ct.anns[ann]['polygon'], np.int32).reshape((4, 2))

            if self.ct.anns[ann]['legibility'] == 'legible':
                # draw only legible bbox/polygon
                cv2.fillConvexPoly(annotation, poly, 255)
            else:
                # 0 weight if it is illegible
                cv2.fillConvexPoly(weight, poly, 0.0)

        bboxes = np.array([
            np.float32(self.ct.anns[ann]['bbox'])
            for ann in self.ct.imgToAnns[coco_id] if ann != []
        ])

        if bboxes.shape[0] == 0:
            return [image, annotation, weight]

        expansion = 0.05
        # Enlarge the bbox by expansion*2% of its size
        # [x, y, w, h] => [x - w*0.05, y - h*0.5, w*1.05, h*1.05]
        bboxes[:,:2] -= bboxes[:,2:] * expansion
        bboxes[:,2:] *= (1 + expansion)

        def intersection(ba, bb):
            new_top = max(ba[1], bb[1])
            new_left = max(ba[0], bb[0])
            new_right = min(ba[0] + ba[2], bb[0] + bb[2])
            new_bottom = min(ba[1] + ba[3], bb[1] + bb[3])
            if new_top < new_bottom and new_left < new_right:
                return [
                    [new_left, new_top],
                    [new_right, new_top],
                    [new_right, new_bottom],
                    [new_left, new_bottom]
                ]
            return None

        for bbox in bboxes:
            for other in bboxes:
                if (bbox != other).all():
                    inter = intersection(bbox, other)
                    if inter is not None:
                        cv2.fillConvexPoly(annotation, np.int32(inter), 127)

        return [image, annotation, weight]

    def _load_image(self, coco_id):
        """
        Load image already saved on the disk
        """
        fname = 'COCO_train2014_%012d.png' % coco_id
        image = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/images/', fname))
        annotation = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/anns/', fname))
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        weight = cv2.imread(
            os.path.join(self.coco_dir, 'subset_validation/weights', fname))
        weight = cv2.cvtColor(weight, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.

        return [image, annotation, weight]

    def _crop_resize(self, image, annotation, weight, name=None):
        # next level hacks
        assert name is not None
        valid_anns = [
            ann for ann in self.ct.imgToAnns[name]
            if self.ct.anns[ann]['legibility'] == 'legible'
        ]
        ann = np.random.choice(valid_anns)
        # extract bbox => x, y, w, h
        bbox_rect = np.int32(self.ct.anns[ann]['bbox'])
        window = coco_utils.get_window(annotation.shape[:2], bbox_rect)
        return coco_utils.crop_resize([image, annotation, weight], window, self.crop_size)
