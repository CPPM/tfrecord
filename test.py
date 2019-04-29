import os
import tensorflow as tf

print(tf.__version__)
filepath = "./image"
pathDir = os.listdir(filepath)
print(pathDir)
img_path = pathDir[0]
print(img_path)

img_raw = tf.gfile.FastGFile(filepath + '/' + img_path, mode='rb').read()
print(repr(img_raw)[:100] + "...")

with tf.Session() as sess:
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)

    print(img_tensor)
    print(img_tensor.dtype)

    img_final = tf.image.resize_images(img_tensor, [192, 192])
    img_final = img_final / 255.0

    print(img_final)
    print(img_final.shape)
    print(tf.reduce_max(img_final).eval())
    print(tf.reduce_min(img_final).eval())


