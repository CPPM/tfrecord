global_step = tf.Variable(
            0, name='global_step', trainable=False)


        # 定义Placeholder，存放输入和标签
        datas_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
        labels_placeholder = tf.placeholder(tf.int32, [None])

        # 存放DropOut参数的容器，训练时为0.25，测试时为0
        dropout_placeholdr = tf.placeholder(tf.float32)

        # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
        conv0 = tf.layers.conv2d(
            datas_placeholder, 10, 3, activation=tf.nn.relu)
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

        # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
        conv1 = tf.layers.conv2d(pool0, 20, 5, activation=tf.nn.relu)
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

        conv2 = tf.layers.conv2d(pool1, 20, 1, activation=tf.nn.relu)

        # 将3维特征转换为1维向量
        flatten = tf.layers.flatten(conv2)

        # 全连接层，转换为长度为100的特征向量
        fc = tf.layers.dense(flatten, 100, activation=tf.nn.relu)

        # 加上DropOut，防止过拟合
        dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

        # 未激活的输出层
        logits = tf.layers.dense(dropout_fc, num_classes)

        correct_prediction_test = tf.equal(
            tf.argmax(logits, 1),
            tf.argmax(tf.one_hot(labels_placeholder, num_classes), 1))

        training_set_accuracy = tf.reduce_mean(
            tf.cast(correct_prediction_test, tf.float32))
        summary_training_set_accuracy = tf.summary.scalar(
            "training_set_accuracy", training_set_accuracy)

        test_set_accuracy = tf.reduce_mean(
            tf.cast(correct_prediction_test, tf.float32))
        summary_test_set_accuracy = tf.summary.scalar(
            "test_set_accuracy", test_set_accuracy)

        # 利用交叉熵定义损失
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels_placeholder, num_classes),
            logits=logits
        )
        # 平均损失
        mean_loss = tf.reduce_mean(losses)
        summary_mean_loss = tf.summary.scalar("mean_loss", mean_loss)

        # worker_device = '/job:worker/task:%d/cpu:1' % FLAGS.task_index
        # with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
        #                                               cluster=cluster
        #                                               )):
        #     定义优化器，指定要优化的损失函数
        optsync = tf.train.SyncReplicasOptimizer(
            tf.train.AdamOptimizer(learning_rate=1e-2),
            replicas_to_aggregate=num_workers-1,
            total_num_replicas=num_workers
        )
        # optsync = tf.train.SyncReplicasOptimizer(
        #     tf.train.AdamOptimizer(learning_rate=1e-2)
        # )