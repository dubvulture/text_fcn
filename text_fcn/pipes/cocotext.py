from __future__ import absolute_import
from __future__ import division
from six.moves import range

from collections import defaultdict
import os
import json
import zipfile

import cv2
import numpy as np
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import labeled_comprehension as extract_feature
from scipy.ndimage.morphology import binary_dilation as dilation

from text_fcn.coco_text import coco_evaluation


def get_bboxes(image, probs):
    """
    Return bounding boxes found and their accuracy score (TBD)
    :param image: B/W image (values in {0, 255})
    :param probs: gray image representing pixelwise confidence
    """
    MIN_AREA = 32
    X = 3
    DIL = (X, X)
    ONES = np.ones(DIL, dtype=np.uint8)

    # pad image in order to prevent closing "constriction"
    output = np.pad(image, (DIL, DIL), 'constant')
    output = dilation(output, structure=ONES, iterations=1).astype(np.uint8)
    # remove padding
    output = output[X:-X, X:-X]
    labels, num = label(output, structure=ONES)
    if num > 0:
        areas = extract_feature(output, labels, range(1, num+1), np.sum, np.int32, 0)

        for i in range(num):
            if areas[i] < MIN_AREA:
                labels[labels == i+1] = 0

        objs = find_objects(labels)

        scores = [
            np.sum((probs[obj][image[obj] > 0]) / 255) / np.count_nonzero(image[obj])
            for obj in objs if obj is not None
        ]
        bboxes = np.array([
            [obj[1].start, obj[1].stop, obj[0].start, obj[0].stop]
            for obj in objs if obj is not None
        ])
    else:
        bboxes, scores = None, None

    return bboxes, scores


def coco_pipe(coco_text, in_dir, mode='validation'):
    """
    :param coco_text: COCO_Text instance
    :param in_dir:
    :param mode:
    """
    directory = os.path.join(in_dir, 'output/')
    fnames = [
        os.path.join(directory, image)
        for image in os.listdir(directory)
    ]
    images = [
        image for image in fnames
        if image.endswith('_output.png')
    ]
    probs = [
        prob for prob in fnames
        if prob.endswith('_scores.png')
    ]
    images.sort()
    probs.sort()

    jsonarr = []
    for i, fname in enumerate(images):
        image = cv2.imread(fname)
        prob = cv2.imread(probs[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prob = cv2.cvtColor(prob, cv2.COLOR_BGR2GRAY)
        # image[image > 0] = 255 (this should not be needed)
        bboxes, scores = get_bboxes(image, prob)
        coco_id = int(fname[-23:-11])

        if bboxes is not None:
            for i in range(bboxes.shape[0]):
                # x1, x2, y1, y2 => x, dx, y, dy
                bboxes[i, 1] -= bboxes[i, 0]
                bboxes[i, 3] -= bboxes[i, 2]
                # x, dx, y, dy => x, y, dx, dy
                bboxes[i] = bboxes[i, [0,2,1,3]]
                jsonarr.append({
                    'utf8_string': "null",
                    'image_id': coco_id,
                    'bbox': bboxes[i].tolist(),
                    "score": scores[i]
                })

    res_dir = os.path.join(directory, 'res/')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(res_dir, 'results.json'), 'w') as f:
        json.dump(jsonarr, f, indent=4)

    if mode == 'validation':
        ct_res = coco_text.loadRes(jsonarr)
        imgIds = [pred['image_id'] for pred in jsonarr]
        detections = coco_evaluation.getDetections(
            coco_text, ct_res, imgIds=None, detection_threshold=0.5)
        coco_evaluation.printDetailedResults(coco_text, detections, None, 'FCN')
    elif mode == 'test':
        with open(os.path.join(directory, 'res/results.json'), 'r') as f:
            jsonarr = json.load(f)

        struct = defaultdict(list)

        for box in jsonarr:
            res = box['bbox']
            res.append(box['score'])
            struct[box['image_id']].append(res)

        files = []
        for coco_id, boxes in struct.items():
            file = os.path.join(directory, 'res_%d.txt' % coco_id)
            files.append(file)
            with open(file, 'w') as f:
                for res in boxes:
                    res = np.array(res)
                    res[[2, 3]] += res[:2]
                    x1, y1, x2, y2, score = res
                    f.write('%d,%d,%d,%d,%f\r\n' % (x1, y1, x2, y2, score))

        for f in files:
            with zipfile.ZipFile(os.path.join(directory, 'res/results.zip'), 'w') as zip:
                zip.write(f)
            os.remove(f)