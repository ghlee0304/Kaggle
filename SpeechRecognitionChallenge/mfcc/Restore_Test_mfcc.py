import tensorflow as tf
import pandas as pd
import numpy as np
import os
import itertools

test_dir_path = "D:/kaggle/DataSet/testnpy/audio"
label_file_path = "D:/kaggle/DataSet/label.csv"
saveDir = "D:/kaggle/Output/FirstTrial/"
ModelDir = saveDir+"model1_99/Model_Adam"
ModelName = "model1"

#Step1 모델 복원 클래스(내부에서 min max scale처리)
class ImportGraph():
    def __init__(self, loc, name):
        self.name = name
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + ".meta",clear_devices=True)
            saver.restore(self.sess, loc)
            self.hypothesis = self.sess.graph.get_tensor_by_name(self.name+"/hypothesis:0")
            self.logits = self.sess.graph.get_tensor_by_name(self.name+"/logits:0")
            self.prediction = self.sess.graph.get_tensor_by_name(self.name+"/prediction:0")

    def run(self, data):
        dvalue = np.expand_dims(data, axis=0)
        return self.sess.run([self.hypothesis, self.logits],feed_dict={self.name+"/X:0": dvalue})
    
def search(dirname):
    filelist = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filelist.append(full_filename.replace('\\', '/'))
    return filelist

#모델 복원
model = ImportGraph(ModelDir,ModelName)

keys=[]
labels=[]
test_labels = ['yes','no','up','down','left','right','on','off','stop','go','silence']
for line in open(label_file_path):
    fields = line.rstrip().split(',')
    if not fields[0] in test_labels:
        keys.append('unknown')
    else:
        keys.append(fields[0])
    label = int(fields[1])
    labels.append(label)

label_dict = dict(itertools.zip_longest(labels,keys))

test_dirs = search(test_dir_path)
total_predict = [['fname','label']]
print('Total number of test samples : ',len(test_dirs))

for i in range(len(test_dirs)):

    a = np.load(test_dirs[i])
    h,l = model.run(a)

    temp = test_dirs[i].split('/')[-1]
    temp = temp.replace('npy','wav')
    b = len(np.where(h>=0.5)[0])
    if b == 0:
        total_predict.append([temp,'silence'])
    else:
        p = np.argmax(h, axis=1)[0]
        total_predict.append([temp,label_dict[p]]) 
    
    if i % 500 == 0:
        print('{}번째 진행 중 >>'.format(i))
        print('예측 : ',total_predict[i+1])
np.savetxt(saveDir+'benzamin_predict5.csv',total_predict,fmt='%s',delimiter=',')

'''
a = np.load('D:/kaggle/DataSet/testnpy/audio/clip_000dcdd2c.npy')
h,l = model.run(a)
print(h)
'''
    
