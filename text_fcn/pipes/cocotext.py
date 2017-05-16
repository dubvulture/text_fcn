from six.moves import range

import os
import json

import numpy as np
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import labeled_comprehension as extract_feature
from scipy.ndimage.morphology import binary_closing as closing

from coco_text import coco_evaluation



def get_bboxes(image):
    """
    Return bounding boxes found and their accuracy score (TBD)
    :param image: B/W image (values in {0, 255})
    """
    MIN_AREA = 32
    X = 3
    DIL = (X, X)
    ONES = np.ones(DIL, dtype=np.uint8)

    # pad image in order to prevent closing "constriction"
    output = np.pad(image, (DIL, DIL), 'constant')
    output = closing(output, structure=ONES, iterations=3).astype(np.uint8)
    # remove padding
    output = output[X:-X, X:-X]
    labels, num = label(output, structure=ONES)
    if num > 0:
        areas = extract_feature(output, labels, range(1, num+1), np.sum, np.int32, 0)

        for i in range(num):
            if areas[i] < MIN_AREA:
                labels[labels == i+1] = 0

        objs = find_objects(labels)

        bboxes = np.array([
            [obj[1].start, obj[1].stop, obj[0].start, obj[0].stop]
            for obj in objs if obj is not None
        ])
        # count white pixels inside the current bbox
        area = lambda b: np.count_nonzero(image[b[2]:b[3], b[0]:b[1]])
        # score as foreground / bbox_area
        scores = [
            #area(bbox) / np.prod(bbox[[3,1]] - bbox[[2,0]])
            1
            for bbox in bboxes
        ]
    else:
        bboxes, scores = None, None

    return bboxes, scores


def coco_pipe(coco_text, in_dir):
    """
    :param coco_text: COCO_Text instance
    """
    directory = os.path.join(in_dir, 'output/')
    fnames = [
        os.path.join(directory, image)
        for image in os.listdir(directory)
    ]
    results = os.path.join(args.logs_dir, 'results.json')
    jsonarr = []
    for fname in fnames:
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image[image > 0] = 255 (this should not be needed)
        bboxes, scores = get_bboxes(image)
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

    with open(results, 'w') as stream:
        json.dump(jsonarr, stream, indent=4)

    ct_res = coco_text.loadRes(jsonarr)
    imgIds = [pred['image_id'] for pred in jsonarr]
    detections = coco_evaluation.getDetections(
        coco_text, ct_res, imgIds=imgIds, detection_threshold=0.5)