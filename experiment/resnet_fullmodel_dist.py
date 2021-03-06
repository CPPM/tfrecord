import os
import time
import logging

import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim import nets

# TODO(limk):将本文件改为分布式，查看读取数据的顺序

print(tf.__version__)

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "172.172.0.2:2232",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "172.172.0.3:2233,172.172.0.4:2234",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", None, "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", None, "Index of task within the job")
tf.app.flags.DEFINE_integer("train_steps", 100, "train_steps of the job")

FLAGS = tf.app.flags.FLAGS
if FLAGS.job_name == 'ps':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if FLAGS.job_name == 'worker':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.task_index)

EPOCHS = 1
BATCH_SIZE = 256
NUM_CLASSES = 1000
LR = 0.01

slim = tf.contrib.slim

data_dir = "/data/train/tfdata/"


ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

num_ps = len(ps_hosts)
num_workers = len(worker_hosts)

LR_scale=LR*num_workers

AUTOTUNE=tf.data.experimental.AUTOTUNE


def input_fn(num_workers=1,index=0):
    # every_worker_dataset_num=int(NUM_CLASSES/num_workers)
    # dataset_begin_index=index*every_worker_dataset_num
    # dataset_end_index=(index+1)*every_worker_dataset_num
    # import pdb;pdb.set_trace()
    train_files_names = os.listdir(data_dir)
    # train_files = [data_dir + item for item in train_files_names[dataset_begin_index:dataset_end_index]]
    train_files = [data_dir + item for item in train_files_names[:1000]]
    dataset_train = tf.data.TFRecordDataset(train_files, buffer_size=2048,
                                            num_parallel_reads=128)

    def _parse_data(example_proto):
        # import pdb;pdb.set_trace()
        features = {'label': tf.FixedLenFeature([], tf.int64),
                    'img_raw': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
        label = tf.cast(parsed_features['label'], tf.int64)
        label = tf.one_hot(label, depth=NUM_CLASSES)
        image = tf.reshape(image, [224, 224, 3])
        image = tf.cast(image, tf.float32) / 255
        return image, label

    dataset_train = dataset_train.repeat(EPOCHS)
    dataset_train = dataset_train.shuffle(buffer_size=1024,seed=1)
    dataset_train = dataset_train.map(_parse_data, num_parallel_calls=AUTOTUNE)
    dataset_train = dataset_train.batch(BATCH_SIZE,drop_remainder=True)
    dataset_train = dataset_train.prefetch(4)
    return dataset_train


def input_fn_test():
    data_image = np.random.random([512, 224, 224, 3])
    data_label = np.random.randint(0, 1000, [512])
    # 生成one_hot编码
    data_label = np.eye(1000)[data_label]
    print(data_image.shape, data_label.shape)

    dataset = tf.data.Dataset.from_tensor_slices((data_image, data_label))

    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.shuffle(buffer_size=4)
    # dataset = dataset.map(_parse_data, num_parallel_calls=30)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(4)
    return dataset


dataset_train = input_fn(num_workers=num_workers,index=FLAGS.task_index)
# dataset_train = input_fn_test()

img_and_label_iter = dataset_train.make_one_shot_iterator().get_next()
#print(img_iter.shape, label_iter.shape)


# with tf.Session() as sess:
#
#
#
#     for i in range(1):
#         print(sess.run(img_iter),sess.run(img_iter).shape)
#         print(sess.run(label_iter),sess.run(label_iter).shape)


# import pdb;
#
# pdb.set_trace()


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    num_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    # gpu_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
        config=gpu_config)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        """
        写日志
        """
        file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'time-and-images-{}-{}-{}.log'.format(FLAGS.job_name,FLAGS.task_index,
                                     str(int(time.time()))))
        logging.basicConfig(filename=file_name, level=logging.INFO)



        is_chief = (FLAGS.task_index == 0)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster,
                ),):
                # ps_strategy=tf.contrib.training.RandomStrategy(num_ps)),):
                # ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(num_ps, tf.contrib.training.byte_size_load_fn)),):
            # jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
            # with jit_scope():

            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
            y = tf.placeholder(tf.int64, [None, 1000])

            """
            model
            """
            net, endpoints = nets.resnet_v1.resnet_v1_50(x,
                                                         num_classes=NUM_CLASSES,
                                                         is_training=None)
            print("1111111111111111111",net.shape)
            net = tf.squeeze(net, axis=[1, 2])
            print("1111111111111111111",net.shape)
            # net = slim.dropout(net, keep_prob=0.5, scope='scope')
            logits = slim.fully_connected(net, num_outputs=NUM_CLASSES,
                                          activation_fn=None, scope='fc')

            # print(net)
            # print("y",y)
            # print("logits",logits)
            # import pdb;pdb.set_trace()

            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.cast(y, tf.float32), logits=logits)
            mean_loss = tf.reduce_mean(losses)

            # logits = tf.nn.softmax(logits)

            # conv0 = tf.layers.Conv2D(filters=4, kernel_size=5)(x)
            #
            # flatten = tf.layers.Flatten()(conv0)
            #
            # y_ = tf.layers.Dense(1000, activation=tf.nn.relu)(flatten)
            #
            # y_ = tf.nn.softmax(y_)
            # print(y_)
            # import pdb;pdb.set_trace()

            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_,
            #                                                     labels=y, )
            # labels = tf.one_hot(
            #     y,
            #     depth=NUM_CLASSES)
            # print("losssss",losses)
            # import pdb;pdb.set_trace()

            # mean_loss = tf.reduce_mean(losses)

            # global_step = tf.Variable(0)

            # global_step = tf.contrib.framework.get_or_create_global_step()
            global_step = tf.train.get_or_create_global_step()
            # print("global_step", global_step)
            # import pdb;
            # pdb.set_trace()

            optsync = tf.train.SyncReplicasOptimizer(
                tf.train.AdamOptimizer(learning_rate=LR),
                replicas_to_aggregate=num_workers,
                total_num_replicas=num_workers,
                use_locking=True
            )

            sync_replicas_hook = optsync.make_session_run_hook(is_chief)
            hooks = [sync_replicas_hook, tf.train.StopAtStepHook(
                last_step=FLAGS.train_steps)]
            train_op = optsync.minimize(mean_loss, global_step=global_step)

            init_op = tf.global_variables_initializer()

            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(
                                                           FLAGS.task_index == 0),
                                                   checkpoint_dir='/home/limk/tfrecord/experiment/log/',
                                                   # save_checkpoint_secs=60
                                                   hooks=hooks,
                                                   config=gpu_config,
                                                   stop_grace_period_secs=60
                                                   ) as mon_sess:
                # 用于保存和载入模型
                # log_dir = FLAGS.log_dir
                mon_sess.run(init_op)
                if is_chief:
                    print(
                        'Worker %d: Initailizing session...' % FLAGS.task_index)
                else:
                    print(
                        'Worker %d: Waiting for session to be initaialized...' %
                        FLAGS.task_index)
                print(
                    'Worker %d: Session initialization complete.' % FLAGS.task_index)

                local_step = 0

                print("sleep...............")
                time.sleep(60)
                # import pdb;
                # pdb.set_trace()
                total_start_time = time.time()
                while not mon_sess.should_stop():
                    # print("labels: {}".format(mon_sess.run(y_iter)))
                    start_time = time.time()
                    try:
                        image_and_label=mon_sess.run(img_and_label_iter)
                        _, loss_, step = mon_sess.run(
                            [train_op, mean_loss, global_step],
                            feed_dict={x: image_and_label[0],
                                       y:image_and_label[1]})
                    except tf.errors.OutOfRangeError:
                        break


                    # _, loss_, step = mon_sess.run(
                    #     [train_op, mean_loss, global_step],
                    #     feed_dict={x: mon_sess.run(img_iter),
                    #                y:
                    #                    mon_sess.run(label_iter)})

                    local_step += 1
                    step_time = time.time() - start_time
                    if step % 10 == 0:
                        images_per_second_in_a_step=BATCH_SIZE / step_time
                        print("local_step: {:<10d}".format(local_step),
                              "global_step: {:<10d}".format(step),
                              "loss_: {:.6f}".format(loss_),
                              " {:.6f} images/s in a step".format(
                                  images_per_second_in_a_step))
                        logging.info('local_step: {:<10d}  step: {:<10d} loss: {:.6f} images_per_second_in_a_step: {:.6f}'.format(local_step, step, loss_,images_per_second_in_a_step))
                total_time = time.time() - total_start_time

                images_per_second=local_step * BATCH_SIZE / total_time

                print("total time: {} s, {:.6f} images/s in a step".format(
                    total_time, images_per_second))
                logging.info('total_time:{},  images/s:{}'.format(total_time, images_per_second))


if __name__ == "__main__":
    tf.app.run()
