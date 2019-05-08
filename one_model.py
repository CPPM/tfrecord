import os

import numpy as np
import tensorflow as tf

print(tf.__version__)
filepath = "./image"

abs_path = os.path.dirname(os.path.abspath(__file__))
print(os.path.abspath(__file__))
print('当前目录绝对路径:', abs_path + '\\image')
image_dir = abs_path + '\\image'

img_name_list = os.listdir(image_dir)
print(img_name_list)

all_image_paths = [image_dir + '\\' + img_name for img_name in img_name_list]
print(all_image_paths)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)


# img_raw = tf.gfile.FastGFile(filepath + '/' + img_path, mode='rb').read()
# print(repr(img_raw)[:100] + "...")


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    image /= 255.0
    return image


def load_and_preprocess_image(path):
    """

    :param path:the path of an image
    :return: a image tensor
    """
    image = tf.read_file(path)
    return preprocess_image(image)


image_ds = path_ds.map(load_and_preprocess_image)
print(image_ds)

all_image_labels = [i for i in range(0, 9)]
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
print(label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
print(ds)


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)

BATCH_SIZE = 1

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=len(all_image_labels))
ds = ds.repeat(1)
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
print(ds)

iterator = ds.make_one_shot_iterator()

x_iter,y_iter = iterator.get_next()

num_classes = 10














x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int64, [None])

conv0 = tf.layers.Conv2D(filters=4, kernel_size=5)(x)

flatten = tf.layers.Flatten()(conv0)

y_ = tf.layers.Dense(10, activation=tf.nn.relu)(flatten)

y_ = tf.nn.softmax(y_)

losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_,
                                                 labels=tf.one_hot(y,
                                                                   depth=num_classes))

mean_loss = tf.reduce_mean(losses)

global_step = tf.Variable(0)

train_op = tf.train.AdagradOptimizer(0.01).minimize(
    losses, global_step=global_step)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    X = np.zeros([20, 224, 224, 3])
    Y = np.zeros([20, 1])
    sess.run(init_op)

    print(sess.run(x_iter))
    print(sess.run(y_iter))
    # TODO(limk):在feed_dict的时候是否可以继续优化

    _,loss_=sess.run([train_op,mean_loss], feed_dict={x: sess.run(x_iter), y: sess.run(y_iter)})
    print(loss_)
