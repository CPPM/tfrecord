import tensorflow as tf

with tf.name_scope("a") as a_scope:
    y = tf.Variable([1.0],name="a")
with tf.name_scope("b") as b_scope:
    y = tf.Variable([1.0],name="a")
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
with tf.Session() as sess:
    pass