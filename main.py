from __future__ import print_function
from six.moves import xrange

import argparse
from datetime import datetime
import os
import subprocess

import tensorflow as tf

from coco_text import COCO_Text
import coco_utils
from dataset_reader import BatchDataset
from text_fcn import text_fcn


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default='1e-04', help='learning rate for Adam Optimizer')
parser.add_argument('--image_size', type=int, default=256, help='image size for training')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--keep_prob', type=float, default=0.85, help='keep probability with dropout')
parser.add_argument('--logs_dir', default='logs/temp/', help='path to logs directory')
parser.add_argument('--coco_dir', default='COCO_Text/', help='path to dataset')
parser.add_argument('--mode', required=True, choices=['train', 'test', 'visualize'])
parser.add_argument('--save_freq', type=int, default=500, help='save model every save_freq')
parser.add_argument('--train_freq', type=int, default=20, help='trace train_loss every train_freq')
parser.add_argument('--val_freq', type=int, default=500, help='trace val_loss every val_freq')
parser.add_argument('--id_list', help='text file containing images\' coco ids to visualize')

args = parser.parse_args()

if args.id_list == None and args.mode == 'visualize':
    parser.error('--mode="visualize" requires --id_list')


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
        f.write('# python main.py')
        for k, v in args._get_kwargs():
            f.write('--%s=\'%s\' ' % (k, v))
        f.write('\n# git checkout master\n\n')


if __name__ == '__main__':
    if not os.path.exists(args.coco_dir):
        raise Exception("coco_dir does not exist")

    args.coco_dir = os.path.abspath(args.coco_dir) + '/'
    args.logs_dir = os.path.abspath(args.logs_dir) + '/'
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
        ckpt = None
    else:
        # Get checkpoint from logs_dir if any
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)

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
        train_set = BatchDataset(train, args.coco_dir, coco_text, opt)
        # We want to keep track of validation loss on a constant dataset
        # => no crops
        val_set = BatchDataset(val, args.coco_dir, coco_text, None)
        # We pass val_set to keep track of its loss
        fcn.train(train_set, val_set=val_set, keep_prob=args.keep_prob)
        
    elif args.mode == 'visualize':
        with open(args.id_list, 'rb') as f:
            ids = [int(line) for line in f if line.strip() != '']
        ids_set = BatchDataset(ids, args.coco_dir, coco_text, None)
        fcn.visualize(ids_set)

    elif args.mode == 'test':
        # We just pass the ids
        fcn.test(test, args.coco_dir)



    
