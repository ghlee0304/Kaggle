import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('./Data/train.csv',
                       header = None, index_col = False, skiprows = 1,
                       names = ['PassengerId','Survived','Pclass,Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
    test = pd.read_csv('./Data/test.csv',
                       header = None, index_col = False, skiprows = 1,
                       names = ['PassengerId','Pclass,Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
    train = train.drop('PassengerId',axis=1)
    target = train['Survived'].values
    train = train.drop('Survived', axis=1)
    data = pd.concat([train,test],axis=0)
    test = test.drop('PassengerId',axis=1)
    ntrain = train.shape[0]
    traindim = train.shape[1]
    ntest = test.shape[0]
    testdim = test.shape[1]
    data =pd.concat([train, test],axis=0)
    data_dummies = pd.get_dummies(data)
    data_array = data_dummies.values
    
    train = data_array[:ntrain,:]
    test = data_array[ntrain:,:]
    target = np.expand_dims(target,1)
    train[np.isnan(train)]=-1
    test[np.isnan(test)]=-1
    np.shape(target)
    ntrain = np.size(train,0)
    ndims = np.size(train,1)
    ntest = np.size(test,0)
    print("The number of train samples : {}".format(ntrain))
    print("The number of test samples : {}".format(ntest))
    print("The number of dims : {}".format(ndims))
    return train, test, target
