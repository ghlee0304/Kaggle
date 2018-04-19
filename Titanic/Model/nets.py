import tensorflow as tf

def fc_relu(_input, output_shape, name, reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable('w', [_input.get_shape()[-1], output_shape], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_shape], initializer = tf.glorot_uniform_initializer())
        h = tf.nn.relu(tf.add(tf.matmul(_input, w),b))
        return h
    
def fc_linear(_input, output_shape, name, reuse = False):
    with tf.variable_scope(name,reuse = reuse):
        w = tf.get_variable('w', [_input.get_shape()[-1], output_shape], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_shape], initializer = tf.glorot_uniform_initializer())
        h = tf.add(tf.matmul(_input, w),b)
        return h

def fc_sigmoid(_input, output_shape, name, reuse = False):
    with tf.variable_scope(name, reuse = reuse):
        w = tf.get_variable('w', [_input.get_shape()[-1], output_shape], initializer = tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [output_shape], initializer = tf.glorot_uniform_initializer())
        h = tf.sigmoid(tf.add(tf.matmul(_input, w),b))
        return h
