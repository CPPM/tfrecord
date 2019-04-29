import os
import tensorflow as tf

print(tf.__version__)
filepath = "/home/limk/tfrecord/image"
pathDir = os.listdir(filepath)
print(pathDir)
img_path =pathDir[0]
print(img_path)

sess=tf.InteractiveSession()

img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_jpeg(img_raw,channels=3)

print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize_images(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final)
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())