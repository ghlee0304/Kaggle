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
Epoch [ 1/30], train loss = 0.234439, train accuracy = 93.52%, valid loss = 0.067374, valid accuracy = 97.88%, duration = 8.235102(s)
Epoch [ 2/30], train loss = 0.081961, train accuracy = 97.40%, valid loss = 0.046846, valid accuracy = 98.33%, duration = 7.440666(s)
Epoch [ 3/30], train loss = 0.056550, train accuracy = 98.24%, valid loss = 0.042627, valid accuracy = 98.58%, duration = 7.265496(s)
Epoch [ 4/30], train loss = 0.041927, train accuracy = 98.71%, valid loss = 0.050592, valid accuracy = 98.40%, duration = 7.408494(s)
Epoch [ 5/30], train loss = 0.028317, train accuracy = 99.08%, valid loss = 0.040519, valid accuracy = 98.93%, duration = 7.226146(s)
Epoch [ 6/30], train loss = 0.021453, train accuracy = 99.32%, valid loss = 0.034781, valid accuracy = 99.03%, duration = 7.315635(s)
Epoch [ 7/30], train loss = 0.014982, train accuracy = 99.50%, valid loss = 0.047708, valid accuracy = 98.98%, duration = 7.477951(s)
Epoch [ 8/30], train loss = 0.011829, train accuracy = 99.61%, valid loss = 0.052679, valid accuracy = 98.75%, duration = 7.355182(s)
Epoch [ 9/30], train loss = 0.009806, train accuracy = 99.69%, valid loss = 0.055246, valid accuracy = 98.90%, duration = 7.208358(s)
Epoch [10/30], train loss = 0.008927, train accuracy = 99.70%, valid loss = 0.040704, valid accuracy = 98.95%, duration = 7.194614(s)
Epoch [11/30], train loss = 0.006031, train accuracy = 99.79%, valid loss = 0.053414, valid accuracy = 98.72%, duration = 7.225066(s)
Epoch [12/30], train loss = 0.004721, train accuracy = 99.83%, valid loss = 0.043436, valid accuracy = 99.07%, duration = 7.213941(s)
Epoch [13/30], train loss = 0.003144, train accuracy = 99.89%, valid loss = 0.047639, valid accuracy = 98.90%, duration = 7.223389(s)
Epoch [14/30], train loss = 0.003773, train accuracy = 99.88%, valid loss = 0.043915, valid accuracy = 99.03%, duration = 7.198126(s)
Epoch [15/30], train loss = 0.003803, train accuracy = 99.89%, valid loss = 0.044459, valid accuracy = 99.08%, duration = 7.220728(s)
Epoch [16/30], train loss = 0.002500, train accuracy = 99.92%, valid loss = 0.050483, valid accuracy = 98.88%, duration = 7.199610(s)
Epoch [17/30], train loss = 0.003352, train accuracy = 99.89%, valid loss = 0.051009, valid accuracy = 98.98%, duration = 7.223580(s)
Epoch [18/30], train loss = 0.001607, train accuracy = 99.95%, valid loss = 0.052006, valid accuracy = 99.03%, duration = 7.163444(s)
Epoch [19/30], train loss = 0.001796, train accuracy = 99.95%, valid loss = 0.036854, valid accuracy = 99.23%, duration = 7.219094(s)
Epoch [20/30], train loss = 0.002072, train accuracy = 99.93%, valid loss = 0.053101, valid accuracy = 99.10%, duration = 7.193491(s)
Epoch [21/30], train loss = 0.002282, train accuracy = 99.94%, valid loss = 0.043718, valid accuracy = 99.15%, duration = 7.206812(s)
Epoch [22/30], train loss = 0.001161, train accuracy = 99.97%, valid loss = 0.041185, valid accuracy = 99.28%, duration = 7.091223(s)
Epoch [23/30], train loss = 0.001275, train accuracy = 99.95%, valid loss = 0.045120, valid accuracy = 99.28%, duration = 7.197087(s)
Epoch [24/30], train loss = 0.000897, train accuracy = 99.98%, valid loss = 0.051325, valid accuracy = 99.03%, duration = 7.208394(s)
Epoch [25/30], train loss = 0.000733, train accuracy = 99.97%, valid loss = 0.054207, valid accuracy = 99.03%, duration = 7.193538(s)
Epoch [26/30], train loss = 0.000482, train accuracy = 99.98%, valid loss = 0.056029, valid accuracy = 99.22%, duration = 7.204233(s)
Epoch [27/30], train loss = 0.000284, train accuracy = 99.99%, valid loss = 0.052089, valid accuracy = 99.15%, duration = 7.206346(s)
Epoch [28/30], train loss = 0.000531, train accuracy = 99.98%, valid loss = 0.050223, valid accuracy = 99.12%, duration = 7.212472(s)
Epoch [29/30], train loss = 0.000468, train accuracy = 99.98%, valid loss = 0.054422, valid accuracy = 99.05%, duration = 7.187961(s)
Epoch [30/30], train loss = 0.000523, train accuracy = 99.98%, valid loss = 0.047944, valid accuracy = 99.28%, duration = 7.212464(s)
Duration for train : 220.898369(s)
<<< Train Finished >>>
Test Accraucy : 99.26%
'''
