import tensorflow as tf
import numpy as np


# # Load the training data into two NumPy arrays, for example using `np.load()`.
# with np.load("/var/data/training_data.npy") as data:
#   features = data["features"]
#   labels = data["labels"]
#
# # Assume that each row of `features` corresponds to the same row as `labels`.
# assert features.shape[0] == labels.shape[0]
#
# features_placeholder = tf.placeholder(features.dtype, features.shape)
# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
#
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# # [Other transformations on `dataset`...]
# dataset = ...
# iterator = dataset.make_initializable_iterator()
#
# sess.run(iterator.initializer, feed_dict={features_placeholder: features,
#                                           labels_placeholder: labels})


data_image=np.random.random([2,224,224,3])
data_label=np.random.randint(0,1000,[2])
# 生成one_hot编码
#data_label=np.eye(1000)[data_label]
print(data_image.shape,data_label.shape)



dataset=tf.data.Dataset.from_tensor_slices((data_image,data_label))

dataset = dataset.repeat(1)
dataset = dataset.shuffle(buffer_size=1024)
# dataset = dataset.map(_parse_data, num_parallel_calls=30)
dataset = dataset.batch(1,drop_remainder=True)
dataset = dataset.prefetch(4)

iterator = dataset.make_one_shot_iterator()
x=iterator.get_next()

# X = tf.placeholder(tf.float16, [None,224,224,3])
# Y = tf.placeholder(tf.int32, [None])
# train_op=y+[1,1]

with tf.Session() as sess:
    # sess.run(iterator.initializer)



    while True:
        try:
            x_=sess.run(x)
            print(x_[0],x_[1])
            # t=sess.run(train_op,feed_dict={X:x,Y:y})
            # print(t)

        except tf.errors.OutOfRangeError:
            break


# input_data=[i for i in range(10)]
#
# dataset=tf.data.Dataset.from_tensor_slices(input_data)
#
# dataset=dataset.shuffle(2,seed=1)
# dataset=dataset.repeat(2)
#
#
# dataset=dataset.batch(1)
#
# iterator=dataset.make_one_shot_iterator()
#
# x=iterator.get_next()
#
# y=x
#
# with tf.Session() as sess:
#
#
#
#     while True:
#         try:
#             print(sess.run(y))
#
#         except tf.errors.OutOfRangeError:
#             break
# print("hkjhkh")

# input_data=[i for i in range(10)]
# inputs=tf.placeholder(tf.float32,[10])
# dataset=tf.data.Dataset.from_tensor_slices(inputs)
#
# dataset=dataset.shuffle(2,seed=1)
# # dataset=dataset.repeat(2)
#
#
# dataset=dataset.batch(1)
#
# iterator=dataset.make_initializable_iterator()
#
# x=iterator.get_next()
#
# data=[i for i in range(10)]
# with tf.Session() as sess:
#
#     sess.run(iterator.initializer,feed_dict={inputs:data})
#
#     while True:
#         try:
#             sess.run(x)
#
#         except tf.errors.OutOfRangeError:
#             break
# print("hkjhkh")



#