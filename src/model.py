import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from time import time
from keras import backend as K

model = Sequential()
model.add(Dense(620, input_shape=(784,)))
model.add(Activation('sigmoid')) 
model.add(Dropout(0.2))  

model.add(Dense(620))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(620))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 