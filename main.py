from __future__ import division, print_function
from six.moves import xrange

import argparse
from datetime import datetime
import json
import os
import subprocess

import cv2
import dill
import numpy as np
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.measurements import label
from scipy.ndimage.measurements import labeled_comprehension as extract_feature
from scipy.ndimage.morphology import binary_closing as closing
import tensorflow as tf

from coco_text import COCO_Text, coco_evaluation
import coco_utils
from dataset_reader import BatchDataset
from text_fcn import text_fcn


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default='1e-04', help='learning rate for Adam Optimizer')
parser.add_argument('--image_size', type=int, default=256, help='image size for training')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--max_steps', type=int, default=0, help='max steps to perform, 0 for infinity')
parser.add_argument('--keep_prob', type=float, default=0.85, help='keep probability with dropout')
parser.add_argument('--logs_dir', default='logs/temp/', help='path to logs directory')
parser.add_argument('--coco_dir', default='COCO_Text/', help='path to dataset')
parser.add_argument('--mode', required=True, choices=['train', 'test', 'visualize', 'coco'])
parser.add_argument('--save_freq', type=int, default=500, help='save model every save_freq')
parser.add_argument('--train_freq', type=int, default=20, help='trace train_loss every train_freq')
parser.add_argument('--val_freq', type=int, default=0, help='trace val_loss every val_freq')
parser.add_argument('--id_list', help='text file containing images\' coco ids to visualize')

args = parser.parse_args()

if args.id_list == None and args.mode == 'visualize':
    parser.error('--mode="visualize" requires --id_list')

assert args.image_size % 32 == 0,\
       'image size must be a multiple of 32'



def save_run():
    """
    Append current run git revision + python arguments to exp.lisp
    :return: git revision string
    """
    git_rev = subprocess.Popen(
        'git rev-parse --short HEAD',
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()[:-1]

    print('Saving run to runs.sh - %s' % git_rev)
    with open('runs.sh', 'a') as f:
        f.write(datetime.now().strftime('# %H:%M - %d %b %y\n'))
        f.write('# git checkout %s\n' % git_rev)
        f.write('# python main.py ')
        for k, v in args._get_kwargs():
            f.write('--%s=\'%s\' ' % (k, v))
        f.write('\n# git checkout master\n\n')


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

        for i in xrange(num):
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


def coco_pipe(coco_text):
    """
    :param coco_text: COCO_Text instance
    """
    directory = os.path.join(args.logs_dir, 'output/')
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
            for i in xrange(bboxes.shape[0]):
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
    coco_evaluation.printDetailedResults(coco_text, detections, None, 'FCN')



if __name__ == '__main__':
    if not os.path.exists(args.coco_dir):
        raise Exception("coco_dir does not exist")

    args.coco_dir = os.path.abspath(args.coco_dir) + '/'
    args.logs_dir = os.path.abspath(args.logs_dir) + '/'
    
    train_set = None
    val_set = None
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
        ckpt = None
    else:
        # Get checkpoint from logs_dir if any
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)
        # And restore train/val sets if they have been serialized
        train_pickle = os.path.join(args.logs_dir, 'train_set.pkl')
        if os.path.exists(train_pickle):
            train_set = dill.load(open(train_pickle, 'rb'))
        val_pickle = os.path.join(args.logs_dir, 'val_set.pkl')
        if os.path.exists(val_pickle):
            val_set = dill.load(open(val_pickle, 'rb'))

    print("Setting up FCN...")
    fcn = text_fcn(
        logs_dir=args.logs_dir,
        lr=args.learning_rate,
        checkpoint=ckpt,
        train_freq=args.train_freq,
        val_freq=args.val_freq,
        save_freq=args.save_freq)

    coco_utils.maybe_download_and_extract(args.coco_dir, coco_utils.URL, is_zipfile=True)
    coco_text = COCO_Text(os.path.join(args.coco_dir, 'COCO_Text.json'))
    train, val, test = coco_utils.read_dataset(coco_text)

    if args.mode == 'train':
        save_run()
        opt = {
            'batch': args.batch_size,
            'size': args.image_size
        }
        train_set = train_set or BatchDataset(train, args.coco_dir, coco_text, opt)
        # We want to keep track of validation loss on an almost constant dataset
        # => load previously saved images/gt/weights
        if args.val_freq > 0:
            subset = os.listdir(os.path.join(
                args.coco_dir, 'subset_validation/images/'))
            subset_ids = [int(i[15:-4]) for i in subset]
            val_set = val_set or BatchDataset(
                subset_ids, args.coco_dir, coco_text, opt, pre_saved=True)

        # We pass val_set (if given) to keep track of its loss
        fcn.train(train_set,
                  val_set=val_set,
                  keep_prob=args.keep_prob,
                  max_steps=args.max_steps)
        
    elif args.mode == 'visualize':
        with open(args.id_list, 'rb') as f:
            ids = [int(line) for line in f if line.strip() != '']
        ids_set = BatchDataset(ids, args.coco_dir, coco_text, None)
        fcn.visualize(ids_set)

    elif args.mode == 'test':
        # We just pass the ids
        fcn.test(test, args.coco_dir)

    elif args.mode == 'coco':
        # After NN extract bboxes and evaluate with coco_text
        perm = np.random.randint(0, len(val), size=[1024])
        val = [coco_text.imgs[coco_id]['file_name'][:-4] for coco_id in val]
        val = np.array(val, dtype=object)[perm].tolist()
        fcn.test(val, args.coco_dir)
        coco_pipe(coco_text)
