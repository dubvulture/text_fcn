from __future__ import print_function
from six.moves import xrange

import argparse
import os
import tensorflow as tf

from dataset_reader import BatchDataset
from coco_text import COCO_Text
import coco_utils
from fcn_net import FCN


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default='1e-04', help='learning rate for Adam Optimizer')
parser.add_argument('--image_size', type=int, default=256, help='image size for training')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--keep_prob', type=float, default=0.85, help='keep probability with dropout')
parser.add_argument('--logs_dir', default='logs/', help='path to logs directory')
parser.add_argument('--coco_dir', default='COCO_Text/', help='path to dataset')
parser.add_argument('--mode', required=True, choices=['train', 'test', 'visualize'])
parser.add_argument('--save_freq', type=int, default=500, help='save model every save_freq')
parser.add_argument('--show_freq', type=int, default=20, help='trace train_loss every show_freq')
parser.add_argument('--val_freq', type=int, default=500, help='trace val_loss every val_freq')


args = parser.parse_args()


if __name__ == '__main__':
    args.coco_dir = os.path.abspath(args.coco_dir) + '/'
    args.logs_dir = os.path.abspath(args.logs_dir) + '/'

    if args.coco_dir is None or not os.path.exists(args.coco_dir):
        raise Exception("coco_dir does not exist")
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    else:
        # Get checkpoint from logs_dir if any
        ckpt = tf.train.get_checkpoint_state(args.logs_dir)

    print("Setting up FCN...")
    fcn = FCN(
        classes=2,
        logs_dir=args.logs_dir,
        lr=args.learning_rate,
        checkpoint=ckpt,
        show_freq=args.show_freq,
        val_freq=args.val_freq,
        save_freq=args.save_freq)

    coco_utils.maybe_download_and_extract(args.coco_dir, coco_utils.URL, is_zipfile=True)
    coco_text = COCO_Text(os.path.join(args.coco_dir, 'COCO_Text.json'))
    train, val, test = coco_utils.read_dataset(coco_text)

    if args.mode == 'train':
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
        val_set = BatchDataset(val, args.coco_dir, coco_text, None)
        FCN.validate(val_set)

    elif args.mode == 'test':
        # We just pass the filenames
        FCN.test(test)



    
