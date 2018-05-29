# -*- coding: utf-8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import time
import load_data

mnist = np.load('./data/kaggle/mnist_train.npy')
x_train = mnist[:,1:]
x_train = np.reshape(x_train, [-1, 28, 28,1])/255.
y_train = mnist[:,[0]]
x_test = np.reshape(np.load('./data/kaggle/test.npy'), [-1, 28, 28, 1])/255.

BOARD_PATH = "./board/kaggle_digit_recognizer_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 100
INIT_LEARNING_RATE = 0.001

ntrain = len(x_train)

image_width = np.size(x_train, 1)
image_height = np.size(x_train, 2)

print("The number of train samples : ", ntrain)


def l1_loss(tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name=name)
    return output


def l2_loss(tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name=name) / 2
    return output


def linear(tensor_op, output_dim, weight_decay=None, regularizer=None, with_W=False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W) * weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W) * weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay=None, regularizer=None,
               keep_prob=1.0, is_training=False, with_W=False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='pre_op')
        bn = tf.contrib.layers.batch_norm(pre_activation,
                                          is_training=is_training,
                                          updates_collections=None)
        h = tf.nn.relu(bn, name='relu_op')
        dr = tf.nn.dropout(h, keep_prob=keep_prob, name='dropout_op')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W) * weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W) * weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return dr, W
        else:
            return dr


def conv2d(tensor_op, stride_w, stride_h, shape, name='Conv'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=shape, dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=shape[-1], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(tensor_op, W, strides=[1, stride_w, stride_h, 1], padding='SAME', name='conv')
    return conv


def max_pooling(tensor_op, ksize_w, ksize_h, stride_w, stride_h, name='MaxPool'):
    with tf.variable_scope(name):
        p = tf.nn.max_pool(tensor_op, ksize=[1, ksize_w, ksize_h, 1], strides=[1, stride_w, stride_h, 1],
                           padding='SAME', name='p')
    return p


def dropout_layer(tensor_op, keep_prob, name):
    with tf.variable_scope(name):
        d = tf.nn.dropout(tensor_op, keep_prob=keep_prob, name = 'd')
    return d


def bn_layer(x, is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(x, updates_collections=None, scale=True, is_training=is_training)
        post_activation = tf.nn.relu(bn, name='relu')
    return post_activation


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

h1 = conv2d(X, 1, 1, [5, 5, 1, 32], name='Conv1')
b1 = bn_layer(h1, is_training, name='bn1')
p1 = max_pooling(b1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
b2 = bn_layer(h2, is_training, name='bn2')
p2 = max_pooling(b2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
b3 = bn_layer(h3, is_training, name='bn3')
p3 = max_pooling(b3, 2, 2, 2, 2, name='MaxPool3')

flat_op = tf.reshape(p3, [-1, 4 * 4 * 128], name='flat_op')
f1 = relu_layer(flat_op, 1024, name='FC_Relu')
d1 = dropout_layer(f1, keep_prob=keep_prob, name='Dropout')
logits = linear(d1, NCLASS, name='FC_Linear')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    predict = tf.argmax(logits, axis=1, name='prediction')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot),
                                name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + tf.reduce_sum(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1)

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis=1)), tf.float32))

with tf.variable_scope("Summary"):
    avg_train_loss = tf.placeholder(tf.float32)
    loss_train_avg = tf.summary.scalar('avg_train_loss', avg_train_loss)
    avg_train_acc = tf.placeholder(tf.float32)
    acc_train_avg = tf.summary.scalar('avg_train_acc', avg_train_acc)
    merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
total_step = int(ntrain / BATCH_SIZE)
print("Total step : ", total_step)
with tf.Session() as sess:
    if os.path.exists(BOARD_PATH):
        shutil.rmtree(BOARD_PATH)
    writer = tf.summary.FileWriter(BOARD_PATH)
    writer.add_graph(sess.graph)

    sess.run(init_op)

    train_start_time = time.perf_counter()
    u = INIT_LEARNING_RATE
    for epoch in range(TOTAL_EPOCH):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))

        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE * step
            t = BATCH_SIZE * (step + 1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],
                                                                   is_training:True,  keep_prob:0.7, learning_rate:u})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        s = sess.run(merged, feed_dict={avg_train_loss: loss_per_epoch, avg_train_acc: acc_per_epoch})
        writer.add_summary(s, global_step=epoch)

        u = u*0.95
        if (epoch + 1) % 1 == 0:
            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    predictions = sess.run(predict, feed_dict={X:x_test, is_training:False, keep_prob:1.0})
    np.savetxt('./data/kaggle/prediction.csv', predictions, delimiter = ',')

'''
GTX 1080Ti
Epoch [ 1/100], train loss = 0.237791, train accuracy = 93.48%, duration = 6.608331(s)
Epoch [ 2/100], train loss = 0.075816, train accuracy = 97.58%, duration = 5.633102(s)
Epoch [ 3/100], train loss = 0.057016, train accuracy = 98.20%, duration = 5.564388(s)
Epoch [ 4/100], train loss = 0.040152, train accuracy = 98.72%, duration = 5.607798(s)
Epoch [ 5/100], train loss = 0.031555, train accuracy = 99.00%, duration = 5.565617(s)
Epoch [ 6/100], train loss = 0.021701, train accuracy = 99.32%, duration = 5.586157(s)
Epoch [ 7/100], train loss = 0.019001, train accuracy = 99.37%, duration = 5.549924(s)
Epoch [ 8/100], train loss = 0.013129, train accuracy = 99.57%, duration = 5.615250(s)
Epoch [ 9/100], train loss = 0.011044, train accuracy = 99.64%, duration = 5.583791(s)
Epoch [10/100], train loss = 0.010262, train accuracy = 99.66%, duration = 5.608551(s)
Epoch [11/100], train loss = 0.007425, train accuracy = 99.77%, duration = 5.615780(s)
Epoch [12/100], train loss = 0.005734, train accuracy = 99.80%, duration = 5.627050(s)
Epoch [13/100], train loss = 0.004804, train accuracy = 99.84%, duration = 5.569445(s)
Epoch [14/100], train loss = 0.004968, train accuracy = 99.87%, duration = 5.593940(s)
Epoch [15/100], train loss = 0.003541, train accuracy = 99.87%, duration = 5.605724(s)
Epoch [16/100], train loss = 0.003491, train accuracy = 99.88%, duration = 5.582156(s)
Epoch [17/100], train loss = 0.002737, train accuracy = 99.90%, duration = 5.618759(s)
Epoch [18/100], train loss = 0.002693, train accuracy = 99.89%, duration = 5.620989(s)
Epoch [19/100], train loss = 0.002163, train accuracy = 99.93%, duration = 5.558762(s)
Epoch [20/100], train loss = 0.002635, train accuracy = 99.91%, duration = 5.632001(s)
Epoch [21/100], train loss = 0.000505, train accuracy = 99.98%, duration = 5.554142(s)
Epoch [22/100], train loss = 0.001908, train accuracy = 99.93%, duration = 5.622885(s)
Epoch [23/100], train loss = 0.001254, train accuracy = 99.95%, duration = 5.649645(s)
Epoch [24/100], train loss = 0.001030, train accuracy = 99.97%, duration = 5.588081(s)
Epoch [25/100], train loss = 0.001424, train accuracy = 99.96%, duration = 5.619541(s)
Epoch [26/100], train loss = 0.001022, train accuracy = 99.96%, duration = 5.604660(s)
Epoch [27/100], train loss = 0.000734, train accuracy = 99.97%, duration = 5.593747(s)
Epoch [28/100], train loss = 0.000154, train accuracy = 100.00%, duration = 5.615468(s)
Epoch [29/100], train loss = 0.000374, train accuracy = 99.99%, duration = 5.565433(s)
Epoch [30/100], train loss = 0.000158, train accuracy = 100.00%, duration = 5.630257(s)
Epoch [31/100], train loss = 0.000614, train accuracy = 99.98%, duration = 5.536249(s)
Epoch [32/100], train loss = 0.000262, train accuracy = 99.99%, duration = 5.555897(s)
Epoch [33/100], train loss = 0.000061, train accuracy = 100.00%, duration = 5.618457(s)
Epoch [34/100], train loss = 0.000497, train accuracy = 99.99%, duration = 5.614348(s)
Epoch [35/100], train loss = 0.000426, train accuracy = 99.99%, duration = 5.566139(s)
Epoch [36/100], train loss = 0.000172, train accuracy = 100.00%, duration = 5.602171(s)
Epoch [37/100], train loss = 0.000355, train accuracy = 99.99%, duration = 5.613516(s)
Epoch [38/100], train loss = 0.000108, train accuracy = 100.00%, duration = 5.549258(s)
Epoch [39/100], train loss = 0.000055, train accuracy = 100.00%, duration = 5.622612(s)
Epoch [40/100], train loss = 0.000064, train accuracy = 100.00%, duration = 5.586841(s)
Epoch [41/100], train loss = 0.000055, train accuracy = 100.00%, duration = 5.567141(s)
Epoch [42/100], train loss = 0.000013, train accuracy = 100.00%, duration = 5.618259(s)
Epoch [43/100], train loss = 0.000212, train accuracy = 100.00%, duration = 5.627549(s)
Epoch [44/100], train loss = 0.000026, train accuracy = 100.00%, duration = 5.552349(s)
Epoch [45/100], train loss = 0.000019, train accuracy = 100.00%, duration = 5.584555(s)
Epoch [46/100], train loss = 0.000022, train accuracy = 100.00%, duration = 5.621929(s)
Epoch [47/100], train loss = 0.000031, train accuracy = 100.00%, duration = 5.597020(s)
Epoch [48/100], train loss = 0.000017, train accuracy = 100.00%, duration = 5.565062(s)
Epoch [49/100], train loss = 0.000015, train accuracy = 100.00%, duration = 5.632291(s)
Epoch [50/100], train loss = 0.000038, train accuracy = 100.00%, duration = 5.603905(s)
Epoch [51/100], train loss = 0.000008, train accuracy = 100.00%, duration = 5.557401(s)
Epoch [52/100], train loss = 0.000003, train accuracy = 100.00%, duration = 5.618168(s)
Epoch [53/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.614196(s)
Epoch [54/100], train loss = 0.000013, train accuracy = 100.00%, duration = 5.562061(s)
Epoch [55/100], train loss = 0.000029, train accuracy = 100.00%, duration = 5.595677(s)
Epoch [56/100], train loss = 0.000005, train accuracy = 100.00%, duration = 5.582001(s)
Epoch [57/100], train loss = 0.000002, train accuracy = 100.00%, duration = 5.585017(s)
Epoch [58/100], train loss = 0.000013, train accuracy = 100.00%, duration = 5.618425(s)
Epoch [59/100], train loss = 0.000004, train accuracy = 100.00%, duration = 5.618253(s)
Epoch [60/100], train loss = 0.000003, train accuracy = 100.00%, duration = 5.614546(s)
Epoch [61/100], train loss = 0.000011, train accuracy = 100.00%, duration = 5.585230(s)
Epoch [62/100], train loss = 0.000021, train accuracy = 100.00%, duration = 5.569173(s)
Epoch [63/100], train loss = 0.000002, train accuracy = 100.00%, duration = 5.625480(s)
Epoch [64/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.602531(s)
Epoch [65/100], train loss = 0.000002, train accuracy = 100.00%, duration = 5.549178(s)
Epoch [66/100], train loss = 0.000002, train accuracy = 100.00%, duration = 5.602425(s)
Epoch [67/100], train loss = 0.000007, train accuracy = 100.00%, duration = 5.619539(s)
Epoch [68/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.544492(s)
Epoch [69/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.607658(s)
Epoch [70/100], train loss = 0.000002, train accuracy = 100.00%, duration = 5.625157(s)
Epoch [71/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.553393(s)
Epoch [72/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.611066(s)
Epoch [73/100], train loss = 0.000003, train accuracy = 100.00%, duration = 5.603319(s)
Epoch [74/100], train loss = 0.000004, train accuracy = 100.00%, duration = 5.561771(s)
Epoch [75/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.612638(s)
Epoch [76/100], train loss = 0.000002, train accuracy = 100.00%, duration = 5.623642(s)
Epoch [77/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.531125(s)
Epoch [78/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.604163(s)
Epoch [79/100], train loss = 0.000004, train accuracy = 100.00%, duration = 5.548570(s)
Epoch [80/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.535531(s)
Epoch [81/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.619305(s)
Epoch [82/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.622139(s)
Epoch [83/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.660935(s)
Epoch [84/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.790956(s)
Epoch [85/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.940266(s)
Epoch [86/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.643531(s)
Epoch [87/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.852271(s)
Epoch [88/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.812818(s)
Epoch [89/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.959436(s)
Epoch [90/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.800464(s)
Epoch [91/100], train loss = 0.000005, train accuracy = 100.00%, duration = 5.603057(s)
Epoch [92/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.779101(s)
Epoch [93/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.561956(s)
Epoch [94/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.595521(s)
Epoch [95/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.709643(s)
Epoch [96/100], train loss = 0.000001, train accuracy = 100.00%, duration = 5.806222(s)
Epoch [97/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.810905(s)
Epoch [98/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.808398(s)
Epoch [99/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.677762(s)
Epoch [100/100], train loss = 0.000000, train accuracy = 100.00%, duration = 5.635224(s)
Duration for train : 564.931560(s)
<<< Train Finished >>>
'''
