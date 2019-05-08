import tensorflow as tf


class MyNetwork(object):
    def __init__(self):
        #创建函数的输入x为数据，y为标签，可以为后续的sess.run的输入
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.int64, [None])

    def my_resnet(self, num_classes=1000, num_workers=2, train_steps=100,
                  is_chief=None):
        """

        :param num_classes:
        :param num_workers:
        :param train_steps:
        :param is_chief:
        :return: mean_loss, hooks, train_op, init_op这些返回值将
        传递给MonitoredTrainingSession函数
        """
        conv0 = tf.layers.Conv2D(filters=4, kernel_size=5)(self.x)

        flatten = tf.layers.Flatten()(conv0)

        y_ = tf.layers.Dense(10, activation=tf.nn.relu)(flatten)

        y_ = tf.nn.softmax(y_)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_,
                                                         labels=tf.one_hot(
                                                             self.y,
                                                             depth=num_classes))

        mean_loss = tf.reduce_mean(losses)

        # global_step = tf.Variable(0)

        global_step = tf.contrib.framework.get_or_create_global_step()

        optsync = tf.train.SyncReplicasOptimizer(
            tf.train.AdamOptimizer(learning_rate=1e-2),
            replicas_to_aggregate=num_workers,
            total_num_replicas=num_workers,
            use_locking=True
        )

        sync_replicas_hook = optsync.make_session_run_hook(is_chief)
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(
            last_step=train_steps)]
        train_op = optsync.minimize(losses, global_step=global_step)

        init_op = tf.global_variables_initializer()

        return mean_loss, hooks, train_op, init_op,global_step

if __name__=='__main__':
    my_net=MyNetwork()
    print(my_net.x)
    print(my_net.y)

    mean_loss, hooks, train_op, init_op=my_net.my_resnet()
    print(mean_loss, hooks, train_op, init_op)