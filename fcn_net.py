from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc

import coco_utils
import TensorflowUtils as utils
from VGG_NET import VGG_NET


class FCN(object):

    def __init__(self,
                 classes,
                 logs_dir,
                 lr=1e-04,
                 checkpoint=None):
        """
        :param classes: # of classes
        :param logs_dir: directory for logs
        :param lr: initial learning rate
        :param checkpoint: a CheckpointState from get_checkpoint_state
        """
        self.classes = classes
        self.logs_dir = logs_dir
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='image')
        self.annotation = tf.placeholder(
            tf.int32, shape=[None, None, None, 1], name='annotation')

        self.prediction, self.logits = self._setup_net()

        self.loss_op = self._loss()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self._training(lr, global_step)

        tf.summary.scalar('train_loss', self.loss_op, collections=['train'])
        tf.summary.scalar('val_loss', self.loss_op, collections=['val'])

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])

        self.summ_train = tf.summary.merge_all(key='train') 
        self.summ_val = tf.summary.merge_all(key='val') 

        self.sv = self._setup_supervisor(checkpoint)    

    def train(self, train_set, val_set=None, keep_prob=0.85):
        """
        :param train_set: training set (get batch of cropped images of same size)
        :param val_set: validation set (in case we want to keep track of its loss)
        """
        with self.sv.managed_session() as sess:
            while not self.sv.should_stop():
                images, anns, _ = train_set.next_batch()
                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.keep_prob: keep_prob}
                sess.run(self.train_op, feed_dict=feed)

                step = sess.run(self.sv.global_step)
                print(step)

                if step % 5 == 0:
                    loss, summary = sess.run(
                        [self.loss_op, self.summ_train],
                        feed_dict=feed)
                    self.sv.summary_computed(sess, summary, step)
                    print('Step: %d\tTrain_loss: %g' % (step, loss))

                if step % 5 == 0:
                    self.sv.saver.save(sess, self.logs_dir + 'model.ckpt', step)
                    if val_set is not None:
                        images, anns, _ = val_set.next_batch()
                        feed = {
                            self.image: images,
                            self.annotation: anns,
                            self.keep_prob: 1.0}
                        # no backpropagation
                        loss, summary = sess.run(
                            [self.loss_op, self.summ_val],
                            feed_dict=feed)
                        self.sv.summary_computed(sess, summary, step/5)
                        print('Step %d\tValidation loss: %g' % (step, loss))

    def validate(self, dataset):
        images, anns, filenames = dataset.get_random_batch()
        feed = {
            self.image: images,
            self.annotation: anns,
            self.keep_prob: 1.0
        }
        pred = sess.run(self.prediction, feed_dict=feed)
        anns = np.squeeze(anns, axis=3)

        for itr in xrange(dataset.image_options['batch']):
            utils.save_image(
                images[itr].astype(np.uint8),
                self.logs_dir,
                name='input_' + str(itr))
            utils.save_image(
                coco_utils.to_mask(anns[itr]),
                self.logs_dir,
                name='gt_' + str(itr))
            utisl.save_image(
                coco_utils.to_mask(pred[itr]),
                self.logs_dir,
                name='pred_' + str(itr))

            print('Saved image: %d' % itr)

    def test(self, filenames):
        for i, fname in enumerate(filenames):
            in_path = os.path.join(self.coco_dir, 'images/', fname+'.jpg')
            in_image = misc.imread(in_path, mode='RGB')
            in_image = np.expand_dims(in_image, axis=0)

            feed = {self.image: in_image, keep_prob: 1.0}
            pred = sess.run(self.prediction, feed_dict=feed)
            print('Evaluated image\t' + fname)

            output = np.squeeze(pred, axis=3)[0]
            utils.save_image(
                coco_utils.to_mask(output),
                self.logs_dir,
                name=fname + '_output')

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
        frac = 1 - tf.nn.zero_fraction(self.annotation)
        weights = tf.squeeze(self.annotation, squeeze_dims=[3])
        weights = tf.one_hot(weights, 1, on_value=1., off_value=frac)
        return tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                logits=self.logits,
                labels=tf.squeeze(self.annotation, squeeze_dims=[3]),
                weights=weights))

    def _setup_net(self):
        """
        Setup the main conv/deconv network
        """
        with tf.variable_scope('inference'):
            vgg_net = VGG_NET(self.image).net
            conv_final = vgg_net['conv5_4']
            output = utils.max_pool_2x2(conv_final)

            conv_shapes = [
                [7, 7, 512, 4096],
                [1, 1, 4096, 4096],
                [1, 1, 4096, self.classes]
            ]

            for i, conv_shape in enumerate(conv_shapes):
                W = utils.weight_variable(
                    conv_shape, name='conv%d_w' % (i+6))
                b = utils.bias_variable(
                    conv_shape[-1:], name='conv%d_b' % (i+6))
                output = utils.conv2d_basic(output, W, b)
                if i<2:
                    output = tf.nn.relu(output, name='relu%d' % (i+6))
                    utils.add_activation_summary(output, collections=['train'])
                    output = tf.nn.dropout(output, keep_prob=self.keep_prob)

            pool4 = vgg_net['pool4']
            pool3 = vgg_net['pool3']

            deconv_shapes = [
                tf.shape(pool4),
                tf.shape(pool3),
                tf.stack([
                    tf.shape(self.image)[0], tf.shape(self.image)[1],
                    tf.shape(self.image)[2], self.classes
                ])
            ]

            W_shapes = [
                [4, 4, pool4.get_shape()[3].value, self.classes],
                [4, 4, pool3.get_shape()[3].value, pool4.get_shape()[3].value],
                [16, 16, self.classes, pool3.get_shape()[3].value]
            ]

            b_shapes = [[shape[2]] for shape in W_shapes]

            strides = [2, 2, 8]

            for i in xrange(3):
                W = utils.weight_variable(
                    W_shapes[i], name='deconv_%d_w' % (i+1))
                b = utils.bias_variable(
                    b_shapes[i], name='deconv_%d_b' % (i+1))
                output = utils.conv2d_transpose_strided(
                    output, W, b,
                    output_shape=deconv_shapes[i], stride=strides[i])
                if i<2:
                    output = tf.add(output, vgg_net['pool%d' % (4-i)])

            prediction = tf.argmax(output, dimension=3, name='prediction')
        return tf.expand_dims(prediction, dim=3), output

    def _setup_supervisor(self, checkpoint):
        """
        Setup the summary writer and variables
        :param checkpoint: saved model if any
        """
        saver = tf.train.Saver(max_to_keep=1)
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
