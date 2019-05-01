import tensorflow as tf

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "172.172.0.2:2232",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "172.172.0.3:2233,172.172.0.4:2234",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

num_classes=10


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  gpu_config = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
  gpu_config.gpu_options.allow_growth = True
  server = tf.train.Server(
      cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,
      config=gpu_config)

  # Create and start a server for the local task.


  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      x=tf.placeholder(tf.float32,[None,224,224,3])
      y=tf.placeholder(tf.int64,[None])
      conv0 = tf.layers.conv2d(
          x, 10, 3, activation=tf.nn.relu)

      flatten = tf.layers.flatten(conv0)

      ft=tf.layers.dense(flatten,10,activation=tf.nn.relu)

      losses=tf.nn.softmax_cross_entropy_with_logits(logits=ft,labels=
      tf.one_hot(y,depth=num_classes))

      mean_loss=tf.reduce_mean(losses)






      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          losses, global_step=global_step)
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             init_op=init_op,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        _, step = sess.run([train_op, global_step])

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
