# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:58:54 2018

@author: shirhe-lyh
"""

# import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import random
import pickle as pk

from tensorflow.contrib.slim import nets

# from tensorflow.contrib.tensorboard.plugins import projector

slim = tf.contrib.slim

flags = tf.flags

flags.DEFINE_string('data_url', '/admin/public/data/cifar/cifar-10-batches-py', 'Path to training images directory.')
flags.DEFINE_string('log_dir', None, '')
flags.DEFINE_string('output_dir', None, 'Path to directory to save model.')
flags.DEFINE_string('checkpoint_path', '/admin/public/data/cifar/cifar-10-batches-py/fine-tune/resnet_v1_50.ckpt', 'Path to the pretrained model.')
# flags.DEFINE_integer('num_images', 10000, 'Number of images to be generated.')
flags.DEFINE_integer('batch_size', 25, 'Batch size.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('num_steps', 10000, 'Number of training steps.')

FLAGS = flags.FLAGS


def unPickle(file):
    with open(file, 'rb') as f:
        data = pk.load(f, encoding='bytes')
    return data


def get_next_batch(batch_size=64):
    """Get a batch set of training data.

    Args:
        batch_size: An integer representing the batch size.
        ...: Additional arguments.

    Returns:
        images: A 4-D numpy array with shape [batch_size, height, width,
            num_channels] representing a batch of images.
        labels: A 1-D numpy array with shape [batch_size] representing
            the groundtruth labels of the corresponding images.

    Raises:
        ValueError: If data_url is not exist.
    """
    #    if not os.path.exists(data_url):
    #        raise ValueError('`data_url` is not exist.')

    i = random.randint(1, 5)
    data = unPickle(FLAGS.data_url + '/data_batch_{}'.format(i))
    images = data.get(b'data')
    labels = data.get(b'labels')
    images = np.reshape(images, (10000, 3, 32, 32))
    images = images.transpose(0, 2, 3, 1)

    #    image_files = np.array(glob.glob(os.path.join(data_url, '*.jpg')))
    #    batch_size = min(batch_size, len(image_files))
    selected_indices = np.random.choice(10000, batch_size)
    selected_images_path = images[selected_indices]
    labels = np.array(labels)
    labels = labels[selected_indices]
    #    for image_path in selected_images_path:
    #        image = cv2.imread(image_path)
    #        image = cv2.resize(image, (224, 224))
    #        label = image_path.split('_')[-1].split('.')[0]
    #        images.append(image)
    #        labels.append(int(label))
    #    images = np.array(images)
    #    labels = np.array(labels)
    return selected_images_path, labels


def main(_):
    # Specify which gpu to be used
    #    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    num_steps = FLAGS.num_steps
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is None:
        raise ValueError('`checkpoint_path` must be specified.')
    model_save_dir = FLAGS.output_dir
    if model_save_dir is None:
        model_save_dir = '/admin/public/model/ResNet_v1_50_finetune/ResNet_finetune_model'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    log_save_dir = FLAGS.log_dir
    if log_save_dir is None:
        log_save_dir = '/admin/public/model/ResNet_v1_50_finetune/ResNet_finetune_log'
    if not os.path.exists(log_save_dir):
        os.mkdir(log_save_dir)

    model_save_path = os.path.join(model_save_dir, 'resnet_finetune.ckpt')


    # data_url = FLAGS.data_url
    # if data_url is None:
    #     data_url = './images'
    # if not os.path.exists(data_url):
    #     os.mkdir(data_url)
    #     generate_train_data.generate_images(FLAGS.num_images, data_url)
    #     print('Generate training images successfully.')

    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    image_shaped_input = tf.reshape(inputs, [-1, 32, 32, 3])
    tf.summary.image('input', image_shaped_input, 10)
    is_training = tf.placeholder(tf.bool, name='is_training')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, num_classes=None,
                                                     is_training=is_training)

    with tf.variable_scope('Logits'):
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        logits = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='fc')
        tf.summary.histogram('net', net)


    checkpoint_exclude_scopes = 'Logits'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [
            scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)

    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(loss)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())

    #    config = tf.ConfigProto(allow_soft_placement = True)
    #    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    #    with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(init)

        # Load the pretrained checkpoint file xxx.ckpt
        saver_restore.restore(sess, checkpoint_path)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_save_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_save_dir + '/test')
        tf.global_variables_initializer().run()

        for i in range(num_steps):
            images, groundtruth_lists = get_next_batch(batch_size)

            # TODO
            # embedding_var = tf.Variable(images, 'data_embedding')
            # config = projector.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = embedding_var.name
            # embedding.metadata = groundtruth_lists
            # projector.visualize_embeddings(tf.summary.FileWriter(log_save_dir), config)

            train_dict = {inputs: images,
                          labels: groundtruth_lists,
                          is_training: True}

            # sess.run(train_step, feed_dict=train_dict)

            # loss_, acc_ = sess.run([loss, accuracy], feed_dict=train_dict)

            # train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
            #     i+1, loss_, acc_)
            # print(train_text)


            if i % 100 == 0:

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                run_metadata = tf.RunMetadata()

                summary, loss_, acc_, _ = sess.run([merged, loss, accuracy, train_step],
                                                   feed_dict=train_dict,
                                                   options=run_options, run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    i + 1, loss_, acc_)
                print(train_text)
            else:
                summary, loss_, acc_, _ = sess.run([merged, loss, accuracy, train_step],
                                                   feed_dict=train_dict)
                train_writer.add_summary(summary, i)
                train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    i + 1, loss_, acc_)
                print(train_text)

            if i == num_steps - 1:
                print('saving...')
                saver.save(sess, model_save_path, global_step=i + 1)
                print('save mode to {}'.format(model_save_dir))

        train_writer.close()


if __name__ == '__main__':
    tf.app.run()
