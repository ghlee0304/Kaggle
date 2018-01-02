import tensorflow as tf
import numpy as np
import random
import time
import os
import itertools
import matplotlib.pyplot as plt

tf.set_random_seed(0)

train_dir_path = "D:/kaggle/DataSet/trainnpy/audio"
test_dir_path = "D:/kaggle/DataSet/testnpy/audio"
label_file_path = "D:/kaggle/DataSet/label.csv"
saveDir = "D:/kaggle/Output/"

def search(dirname):
    filelist = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename.replace('\\', '/'))
    return filelist

train_dirs = search(train_dir_path)
trainX = []
validationX = []
testX = []
nclass = len(train_dirs)
print("클래스의 개수 : ",nclass)

random.seed(0)
for class_dir in train_dirs:
    audio_dirs = search(class_dir)
    naudio = len(audio_dirs)
    ntrain = int(naudio*0.7)
    nvalidation = int(naudio*0.1)
    ntest = int(naudio-ntrain-nvalidation)
    random.shuffle(audio_dirs)
    for i in range(naudio):
        trainX.append(audio_dirs[i])

total_train_num = len(trainX)
print("The number of train samples : ",total_train_num)

keys=[]
labels=[]
for line in open(label_file_path):
    fields = line.rstrip().split(',')
    keys.append(fields[0])
    label = int(fields[1])
    labels.append(label)

label_dict = dict(itertools.zip_longest(keys,labels))

def zero_pad(A, max_length):
    arr = np.zeros([max_length,30])
    arr[:len(A)] = A
    return arr

def make_batch(data, label_dict):
    nbatch = len(data)
    _max = 0
    tdata = []
    for i in range(nbatch):
        a = data[i]
        a = np.load(a)
        tdata.append(a)
        if _max<len(a):
            _max = len(a)
    max_length = _max
    for i in range(nbatch):
        a = data[i]
        temp_label = a.split('/')[-2]
        temp_label = np.array([[label_dict[temp_label]]])
        temp_feature = tdata[i]
        temp_feature = zero_pad(temp_feature,max_length)
        temp_feature = np.expand_dims(temp_feature,axis=0)
        if i == 0:
            features = temp_feature
            labels = temp_label
        else:
            features = np.append(features, temp_feature, axis=0)
            labels = np.append(labels, temp_label,axis=0)
    return features, labels

class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()
        
    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length
    
    def _last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        output_size = tf.shape(output)[2]
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant
    
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, None, 30], name='X')
            self.Y = tf.placeholder(tf.int32, [None, 1], name='Y')
            self.learning_rate = tf.placeholder(tf.float32,name = 'learning_rate')
            self.Y_one_hot = tf.reshape(tf.one_hot(self.Y, 30), [-1, 30])
            len_X=self.length(self.X)

            cell = tf.contrib.rnn.LSTMCell(num_units = 200, initializer = tf.glorot_uniform_initializer(seed=0), state_is_tuple=True)
            outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, sequence_length=len_X)
            last = self._last_relevant(outputs, len_X)

            W1 = tf.Variable(tf.truncated_normal([200, 30], seed = 0, stddev=0.01),name='W1')
            b1 = tf.Variable(tf.constant(0.001, shape=[30]),name='b1')
            self.logits = tf.add(tf.matmul(last,W1),b1,name='logits')
            self.hypothesis = tf.nn.softmax(self.logits, name='hypothesis')
            self.cost = -tf.reduce_mean(self.Y_one_hot * tf.log(self.hypothesis), name='cost')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

            self.prediction = tf.argmax(self.hypothesis, 1, name='prediction')
            self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y_one_hot, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

    def prediction(self, x_test):
        return self.sess.run(self.prediction, feed_dict = {self.X:x_test})

    def train(self, x_train, y_train, u):
        return sess.run([self.accuracy, self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.learning_rate: u})

    def get_accuracy(self, x_test, y_test):
        _n = len(x_test)
        _a,_c = sess.run([self.accuracy,self.cost], feed_dict = {self.X:x_test, self.Y:y_test})
        return _a/_n, _c

    def save(self, dirname):
        saver = tf.train.Saver()
        saver.save(self.sess,dirname)

sess = tf.Session()
model_name = 'model1'
m = Model(sess, model_name)

init = tf.global_variables_initializer()
sess.run(init)

total_epochs = 100
batch_size = 200
total_step = int(total_train_num/batch_size)
train_size = total_step*batch_size
init_learning_rate = 0.001
u = init_learning_rate 
trainX = np.array(trainX)
print("Learning start")
print("total_step : ",total_step)
for epoch in range(total_epochs):
    np.random.seed(epoch)
    mask = np.random.permutation(total_train_num)
    trainX = trainX[mask]
    avg_acc = 0
    avg_cost = 0
    u = u*0.99**epoch
    for step in range(total_step):
        if epoch == 0 and step == 0:
            startTime = time.perf_counter()
        data = trainX[step*batch_size:(step+1)*batch_size]
        batchX, batchY = make_batch(data, label_dict)
        a, c, _ = m.train(batchX, batchY, u)
        avg_acc += a
        avg_cost +=c
        if epoch == 0 and step == 0:
            endTime = time.perf_counter()
            print("1 Step 걸리는 시간(s) : {:.6f}".format(endTime-startTime))
        if step%100== 0:            
            print("Step : {}, accuracy : {:.2%}, cost : {:.6f}".format(step, a/batch_size, c))
    avg_acc = avg_acc / (total_step*batch_size)
    avg_cost = avg_cost / total_step
    print("Epoch : {}, train accuracy : {:.2%}, train cost : {:.6f}\n".format(epoch, avg_acc, avg_cost))
    '''
    data = validationX
    batchX, batchY = make_batch(data, label_dict)
    a, c = m.get_accuracy(batchX, batchY)
    print("Epoch : {}, validation accuracy : {:.2%}, validation cost : {:.6f}\n".format(epoch, a, c))
    '''
    m.save(saveDir+model_name+"_{}/Model_Adam".format(epoch))
