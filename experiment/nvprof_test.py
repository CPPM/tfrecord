import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] ="0"
with tf.device("/device:GPU:0"):
    a=tf.Variable(tf.constant(1))
    b=tf.Variable(tf.constant(1))
    c=a+b

gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
init_op=tf.global_variables_initializer()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    sess.run(init_op)
    print(sess.run(c))