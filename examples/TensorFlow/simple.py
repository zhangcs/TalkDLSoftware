import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('input'):
        a = tf.constant(1, name='a')
        b = tf.constant(2, name='b')

    with tf.name_scope('output'):
        c = tf.multiply(a, b)
        tf.summary.histogram('c', c)
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./board', graph)

with tf.Session(graph=graph) as sess:
    result = sess.run(c)
    print result
    writer.add_summary(merged.eval())

writer.close()
