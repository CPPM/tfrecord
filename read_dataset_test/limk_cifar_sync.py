# coding=utf-8

import os

import tensorflow as tf

flags = tf.app.flags



flags.DEFINE_integer('train_steps', 10,
                     'Number of training steps to perform')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '172.17.0.2:2333',
                    'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '172.17.0.3:2333,172.17.0.4:2333',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1

num_classes=10


def main(argv=None):
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_spec)

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    gpu_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
        config=gpu_config)
    if FLAGS.job_name == u'ps':
        server.join()



    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task:%d/gpu:%d' % (FLAGS.task_index, FLAGS.task_index)
    worker_device = '/job:worker/task:%d/gpu:0' % (FLAGS.task_index)

    print('====================================================')
    print('worker_device: %s' % worker_device)
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                  cluster=cluster
                                                  )):
        # with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        global_step = tf.Variable(
            0, name='global_step', trainable=False)


        optsync = tf.train.SyncReplicasOptimizer(
            tf.train.AdamOptimizer(learning_rate=1e-2),
            replicas_to_aggregate=num_workers-1,
            total_num_replicas=num_workers
        )
        # optsync = tf.train.SyncReplicasOptimizer(
        #     tf.train.AdamOptimizer(learning_rate=1e-2)
        # )
        sync_replicas_hook = optsync.make_session_run_hook(is_chief)
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(
            last_step=FLAGS.train_steps)]
        optimizer = optsync.minimize(losses, global_step=global_step)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(
                                                       FLAGS.task_index == 0),
                                               hooks=hooks,
                                               config=gpu_config,
                                               save_checkpoint_secs=60) as mon_sess:
            # 用于保存和载入模型
            log_dir = FLAGS.log_dir
            if is_chief:
                print('Worker %d: Initailizing session...' % FLAGS.task_index)
            else:
                print('Worker %d: Waiting for session to be initaialized...' %
                      FLAGS.task_index)
            print(
                'Worker %d: Session initialization complete.' % FLAGS.task_index)

            local_step = 0
            step = 0
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                # 训练模型

                if is_chief:
                    # write = tf.summary.FileWriter(
                    #     log_dir, sess.graph)
                    pass

                print("now pre_index is {}".format(pre_index))



                # g = sess.run(global_step)
                # print('zhaorenming global_step:%d' %g)
                print(
                    'Worker %d: training local count %d done (global step:%d)' %
                    (FLAGS.task_index, local_step, step))

                local_step += 1
                # if step>=FLAGS.train_steps:
                #     break

            print("training is over,save model into:{}".format(log_dir))


if __name__ == "__main__":
    tf.app.run()
