from __future__ import absolute_import
from __future__ import print_function
from six.moves import range

import os
import sys

import cv2
import dill
import tensorflow as tf
import numpy as np

from text_fcn import tf_utils
from text_fcn.networks import create_fcn


class TextFCN(object):

    def __init__(self,
                 logs_dir,
                 lr=1e-04,
                 checkpoint=None,
                 train_freq=10,
                 val_freq=0,
                 save_freq=500):
        """
        :param logs_dir: directory for logs
        :param lr: initial learning rate
        :param checkpoint: a CheckpointState from get_checkpoint_state
        :param train_freq: trace train_loss every train_freq
        :param val_freq: trace val_loss every val_freq
        :param save_freq: save model every save_freq
        """
        self.logs_dir = logs_dir
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='image')
        self.annotation = tf.placeholder(
            tf.int32, shape=[None, None, None, 1], name='annotation')
        self.weight = tf.placeholder(
            tf.float32, shape=[None, None, None, 1], name='weight')

        self.prediction, self.logits = create_fcn(self.image, self.keep_prob, 2)

        self.score = tf.nn.softmax(self.logits)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss_op = self._loss()
        self.train_op = self._training(lr, global_step)

        tf.summary.scalar('train_loss', self.loss_op, collections=['train'])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])

        self.summ_train = tf.summary.merge_all(key='train')
        self.summ_val = tf.Summary()

        self.sv = self._setup_supervisor(checkpoint)

        self.train_freq = train_freq
        self.save_freq = save_freq
        self.val_freq = val_freq

    def train(self, train_set, val_set=None, keep_prob=0.85, max_steps=0):
        """
        :param train_set: training set (cropped images of same size)
        :param val_set: validation set (to keep track of its loss)
        :param keep_prob: 1-dropout
        :param max_steps: max steps to perform
        """
        with self.sv.managed_session() as sess:
            print('Starting training...')
            while not self.sv.should_stop():
                images, anns, weights, _ = train_set.next_batch()
                # Transform to match NN inputs
                images = images.astype(np.float32) / 255.
                anns = anns.astype(np.int32) // 255
                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.weight: weights,
                    self.keep_prob: keep_prob
                }
                sess.run(self.train_op, feed_dict=feed)

                step = sess.run(self.sv.global_step)

                if (step == max_steps) or ((step % self.train_freq) == 0):
                    loss, summary = sess.run(
                        [self.loss_op, self.summ_train],
                        feed_dict=feed)
                    self.sv.summary_computed(sess, summary, step)
                    print('Step %d\tTrain_loss: %g' % (step, loss))

                if ((val_set is not None) and (self.val_freq > 0) and
                        (((step % self.val_freq) == 0) or (step == max_steps))):
                    # Average loss on whole validation (sub)set
                    iters = val_set.size // val_set.batch_size
                    mean_loss = 0
                    for i in range(iters):
                        print('Running validation... %d/%d' % (i+1, iters), end='\r')
                        sys.stdout.flush()
                        images, anns, weights, _ = val_set.next_batch()
                        # Transform to match NN inputs
                        images = images.astype(np.float32) / 255.
                        anns = anns.astype(np.int32) // 255
                        feed = {
                            self.image: images,
                            self.annotation: anns,
                            self.weight: weights,
                            self.keep_prob: 1.0
                        }
                        # no backpropagation
                        loss = sess.run(self.loss_op, feed_dict=feed)
                        mean_loss += loss

                    self.summ_val.value.add(tag='val_loss', simple_value=mean_loss/iters)
                    self.sv.summary_computed(sess, self.summ_val, step)
                    self.summ_val.value.pop()
                    print('\nStep %d\tValidation loss: %g' % (step, mean_loss / iters))

                if (step == max_steps) or ((self.save_freq > 0) and
                                           (step % self.save_freq) == 0):
                    # Save model
                    self.sv.saver.save(sess, self.logs_dir + 'model.ckpt', step)
                    print('Step %d\tModel saved.' % step)
                    # Save train & set dataset state
                    dill.dump(
                        train_set,
                        open(os.path.join(self.logs_dir, 'train_set.pkl'), 'wb'))
                    dill.dump(
                        val_set,
                        open(os.path.join(self.logs_dir, 'val_set.pkl'), 'wb'))

                if step == max_steps:
                    break

    def test(self, filenames, directory):
        """
        Run on images in directory without their groundtruth
        (hence this should be used only during validation/testing phase)
        :param filenames:
        :param directory:
        """
        with self.sv.managed_session() as sess:
            for i, fname in enumerate(filenames):
                in_path = os.path.join(directory, fname)
                in_image = cv2.imread(in_path + '.jpg')
                # pad image to the nearest multiple of 32
                dy, dx = tf_utils.get_pad(in_image)
                in_image = tf_utils.pad(in_image, dy, dx)
                # batch size = 1
                in_image = np.expand_dims(in_image, axis=0)
                in_image = in_image.astype(np.float32) / 255.

                feed = {self.image: in_image, self.keep_prob: 1.0}
                pred, score = sess.run([self.prediction, self.score], feed_dict=feed)
                print('Evaluated image\t' + fname)

                # squeeze dims and undo padding
                dy = pred.shape[1] - dy
                dx = pred.shape[2] - dx
                output = np.squeeze(pred, axis=(0,3))[:dy, :dx]
                score = np.squeeze(score, axis=0)[:dy, :dx, 1]
                out_dir = os.path.join(self.logs_dir, 'output/')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                tf_utils.save_image(
                    (output * 255).astype(np.uint8),
                    out_dir,
                    name=fname + '_output')
                tf_utils.save_image(
                    (score * 255).astype(np.uint8),
                    out_dir,
                    name=fname + '_scores')

    def visualize(self, vis_set):
        """
        Run on given images in order to save input & gt & prediction
        """
        with self.sv.managed_session() as sess:
            while True:
                images, anns, _, coco_ids = vis_set.next_batch()

                if vis_set.epoch != 1:
                    # Just one iteration over all dataset's images
                    break

                # pad single image to the nearest multiple of 32
                dy, dx = tf_utils.get_pad(images[0])
                images = tf_utils.pad(images[0], dy, dx)
                anns = tf_utils.pad(anns[0], dy, dx)
                #weights = tf_utils.pad(weights[0], dy, dx, val=1)
                # need to expand again batch size dimension
                images = np.expand_dims(images, axis=0)
                anns = np.expand_dims(anns, axis=0)
                #weights = np.expand_dims(weights, axis=0)

                # Transform to match NN inputs
                images = images.astype(np.float32) / 255.
                anns = anns.astype(np.int32) // 255
                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.keep_prob: 1.0
                }
                preds, score = sess.run([self.prediction, self.score], feed_dict=feed)
                # squeeze dims and undo padding
                dy = preds.shape[1] - dy
                dx = preds.shape[2] - dx
                preds = np.squeeze(preds, axis=(0,3))[:dy, :dx]
                anns = np.squeeze(anns, axis=(0,3))[:dy, :dx]
                images = np.squeeze(images, axis=0)[:dy, :dx]
                coco_ids = np.squeeze(coco_ids, axis=0)

                out_dir = os.path.join(self.logs_dir, 'visualize/')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                tf_utils.save_image(
                    (images * 255).astype(np.uint8),
                    out_dir,
                    name='input_%05d' % coco_ids)
                tf_utils.save_image(
                    (anns * 255).astype(np.uint8),
                    out_dir,
                    name='gt_%05d' % coco_ids)
                tf_utils.save_image(
                    (preds * 255).astype(np.uint8),
                    out_dir,
                    name='pred_%05d' % coco_ids)
                tf_utils.save_image(
                    (score[0,:,:,1] * 255).astype(np.uint8),
                    out_dir,
                    name='prob_%05d' % coco_ids)

                print('Saved image: %d' % coco_ids)
                score = np.mean(np.abs(score[:,:,:,0] - score[:,:,:,1]), axis=(1,2))
                print('Score: %g' % score[0])

    def _training(self, lr, global_step):
        """
        Setup the training phase
        :param lr: initial learning rate
        :param global_step: global step of training
        """
        optimizer = tf.train.AdamOptimizer(lr)
        grads = optimizer.compute_gradients(self.loss_op)
        return optimizer.apply_gradients(grads, global_step=global_step)

    def _loss(self):
        """
        Setup the loss function
        """
        return tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                logits=self.logits,
                labels=self.annotation,
                weights=self.weight))

    def _setup_supervisor(self, checkpoint):
        """
        Setup the summary writer and variables
        :param checkpoint: saved model if any
        """
        saver = tf.train.Saver(max_to_keep=20)
        sv = tf.train.Supervisor(
            logdir=self.logs_dir,
            save_summaries_secs=0,
            save_model_secs=0,
            saver=saver)

        # Restore checkpoint if given
        if checkpoint and checkpoint.model_checkpoint_path:
            print('Checkpoint found, loading model')
            with sv.managed_session() as sess:
                sv.saver.restore(sess, checkpoint.model_checkpoint_path)

        return sv
