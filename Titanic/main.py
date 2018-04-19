import tensorflow as tf
from titanic_utils import*
import numpy as np 
from Model import classifer

train, test, target = load_data()
input_dim = np.size(train,1)
target_dim = np.size(target,1)

with tf.Session() as sess:
    logistic = classifer.LogisticClassifier(sess)
    logistic.build_model(input_dim, target_dim, False)
    logistic.train(0.001, train, target, 11, 32)
    a = logistic.get_predict(test)
    np.savetxt('./Output/predict.csv',a,delimiter = ',')

