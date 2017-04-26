import tensorflow as tf

# Define a computational graph for TensorFlow
graph = tf.Graph()

# Define operation nodes in the graph
with graph.as_default():

    with tf.name_scope('input'):
        # Define two constant symbols (two nodes)
        a = tf.constant(1, name='a')
        b = tf.constant(2, name='b')

    with tf.name_scope('output'):
        # Define an operation node: c = a * b
        c = tf.multiply(a, b)
        tf.summary.histogram('c', c)
    
    # Merge all summaries and write to logdir
    logdir = '/tmp/tensorboard'
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, graph)

with tf.Session(graph=graph) as sess:
    # Compute and output the result (c=a*b)
    result = sess.run(c)
    print result
    
    # Write TensorBoard data to logdir
    writer.add_summary(merged.eval())

writer.close()
