import os

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

BATCH_SIZE = 3

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=len(all_image_labels))
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
print(ds)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=3,kernel_size=5,input_shape=[224,224,3],padding="same"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(all_image_labels))
])

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.fit(ds, epochs=1, steps_per_epoch=3)



