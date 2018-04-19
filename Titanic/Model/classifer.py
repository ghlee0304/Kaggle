import tensorflow as tf
import numpy as np 
from .nets import *

class LogisticClassifier(object):
    tf.set_random_seed(0)
    def __init__(self, sess):
        self.sess = sess

    def build_model(self, input_dim, target_dim, reuse):
        self.X = tf.placeholder(shape = [None, input_dim],dtype = tf.float32)
        self.Y = tf.placeholder(shape = [None,target_dim], dtype = tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.beta1 = 0.5
        
        h1 = fc_relu(self.X, 1000,'hidden1', reuse)
        h2 = fc_relu(h1, 500,'hidden2', reuse)
        h3 = fc_sigmoid(h2, 1, 'output',reuse)
    
        with tf.variable_scope('Adam',reuse = reuse):
            self.loss = tf.reduce_sum(tf.square(h3-self.Y))
            self.optim = tf.train.AdamOptimizer(self.learning_rate,  beta1 = self.beta1).minimize(self.loss)
            
        self.prediction = tf.cast(h3 > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.Y), dtype=tf.float32))
        
    def train(self, learning_rate, trainset,target, total_epoch, batch_size):
        tf.global_variables_initializer().run()
        ntrain = np.size(trainset,0)
        total_step = ntrain//batch_size
        u = learning_rate
        for epoch in range(total_epoch):
            np.random.seed(epoch)
            mask = np.random.permutation(ntrain)
            sum_acc = 0
            sum_loss = 0 
            for step in range(total_step):
                s = step*batch_size
                t = (step+1)*batch_size
                batch_x = trainset[mask[s:t],:]
                batch_y = target[mask[s:t],:]
                _, _loss, _acc = self.sess.run([self.optim, self.loss, self.accuracy], 
                                             feed_dict={self.X : batch_x, self.Y : batch_y, self.learning_rate : u})    
                sum_acc+=_acc
                sum_loss+=_loss
            print("Epoch {}, accuracy : {:.2%}, loss : {:.6f}".format(epoch, sum_acc/total_step, sum_loss/total_step))
            u = u*0.97
    def get_predict(self, testset):
        return self.sess.run(self.prediction,feed_dict={self.X:testset})
