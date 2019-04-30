import numpy as np
import tensorflow as tf

num_classes = 10

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int64, [None, 1])
conv0 = tf.layers.conv2d(
    x, 10, 3, activation=tf.nn.relu)

flatten = tf.layers.flatten(conv0)

ft = tf.layers.dense(flatten, 10, activation=tf.nn.relu)

losses = tf.nn.softmax_cross_entropy_with_logits(logits=ft, labels=
tf.one_hot(y, depth=num_classes))

mean_loss = tf.reduce_mean(losses)

global_step = tf.Variable(0)

train_op = tf.train.AdagradOptimizer(0.01).minimize(
    losses, global_step=global_step)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    X = np.zeros([20, 224, 224, 3])
    Y = np.zeros([20, 1])

    sess.run(train_op, feed_dict={x: X, y: Y})
    print(mean_loss)
