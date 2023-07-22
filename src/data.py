# import necessary libaraies for project
import numpy as np
import pandas as pd

# import csv data with pandas to project
train = pd.read_csv('/home/rezaz/deeplearningProject/MLP/Data/fashion-mnist_train.csv')
test = pd.read_csv('/home/rezaz/deeplearningProject/MLP/Data/fashion-mnist_test.csv')

# extract label of data from train and test
train_labels = np.array(train.label)
test_labels = np.array(test.label)

# drop label colomn of train and test 
train.drop('label',axis=1, inplace=True)
test.drop('label',axis=1, inplace=True)

# split train to x_train and X_valid and convert it to float array 
X_train = train[10000:]
y_train = train_labels[10000:]
X_train = np.array(X_train)
X_train = X_train.astype('float32')

# split train to x_train and X_valid and convert it to float array
X_valid = train[:10000]
y_valid = train_labels[:10000]
X_valid = np.array(X_valid)
X_valid=X_valid.astype('float32')

# convert test data to float array
X_test = test
y_test = test_labels
X_test = np.array(X_test)
X_test = X_test.astype('float32')

# in the above we convert data to float then we can divide data with 255 and this is help us to scale data and learning process is optimised
X_train /= 255
X_test /= 255
X_valid /= 255

