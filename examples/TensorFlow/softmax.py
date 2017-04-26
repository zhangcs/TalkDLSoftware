#-*-coding:utf-8-*-

import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

graph = tf.Graph()

# 下面大量使用 tf.name_scope() 可以使在浏览器中得到 computational graph 更形象更容易阅读
with graph.as_default():
    with tf.name_scope('input_features'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
    with tf.name_scope('input_labels'):
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    with tf.name_scope('input_images'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 20)

    with tf.name_scope('parameters'):
        W = tf.Variable(tf.zeros([784, 10]), name='weights')
        tf.summary.histogram('WEIGHTS', W)
        b = tf.Variable(tf.zeros([10]), name='biases')
        tf.summary.histogram('BIASES', b)
    
    with tf.name_scope('use_softmax'):
        y = tf.nn.softmax(tf.matmul(x, W) + b)

    with tf.name_scope('train'):
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    with tf.name_scope('test'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('Accuracy', accuracy)

    #merged = tf.summary.merge([accuracy_summary])
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./board', graph)

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for step in range(1000):
        if (step%10) == 0:
            feed = {x: mnist.test.images, y_: mnist.test.labels}
            _, acc = sess.run([merged, accuracy], feed_dict=feed)
            print('Accuracy at %s step: %s' % (step, acc))
        else:
            batch_x, batch_y = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
            writer.add_summary(merged.eval(feed_dict={x: batch_x, y_: batch_y}), global_step=step)
    writer.close()
