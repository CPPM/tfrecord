import tensorflow as tf
input_data=[i for i in range(10)]

dataset=tf.data.Dataset.from_tensor_slices(input_data)

dataset=dataset.shuffle(2,seed=1)
dataset=dataset.repeat(2)


dataset=dataset.batch(1)

iterator=dataset.make_one_shot_iterator()

x=iterator.get_next()

y=x

with tf.Session() as sess:



    for i in range(2*len(input_data)):
        print(sess.run(x))
print()