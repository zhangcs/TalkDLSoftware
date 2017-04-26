# Do NOT show warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import TensorFlow
import tensorflow as tf

# Import training and test data from MNIST
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

graph = tf.Graph()

with graph.as_default():

    # Nodes can be grouped into visual blocks for TensorBoard
    with tf.name_scope('input_features'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')

    with tf.name_scope('input_labels'):
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    with tf.name_scope('parameters'):
        W = tf.Variable(tf.zeros([784, 10]), name='weights')
        b = tf.Variable(tf.zeros([10]), name='biases')
        tf.summary.histogram('WEIGHTS', W)
        tf.summary.histogram('BIASES', b)
    
    with tf.name_scope('use_softmax'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)

    with tf.name_scope('train'):
        # Compute the cross entropy of real label y_ and prediction label y
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        # Create a gradient-descent optimizer with learning rate = 0.01
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Track accuracy over time for TensorBoard
        tf.summary.scalar('Accuracy', accuracy)

    logdir = '/tmp/tensorboard'  # temporary path for storing TB summaries
    merged = tf.summary.merge_all()  # Merge all the summaries
    writer = tf.summary.FileWriter(logdir, graph)  # Write summaries

with tf.Session(graph=graph) as sess:
    # Initialize all variables
    tf.global_variables_initializer().run()
    
    for step in range(1,501):
        if (step%10) == 0:
            feed = {x: mnist.test.images, y_: mnist.test.labels}
            _, acc = sess.run([merged, accuracy], feed_dict=feed)
            print('Accuracy at %s step: %s' % (step, acc))
        else:
            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
            writer.add_summary(merged.eval(feed_dict={x: batch_x, y_: batch_y}), global_step=step)

    writer.close()

print("Run the command line to start TensorBoard:\n" \
      "(TensorFlow) $ tensorboard --logdir=/tmp/tensorboard" \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
