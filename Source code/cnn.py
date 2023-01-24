# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:28:49 2021

@author: sas11
"""
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization,Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D,LSTM,Bidirectional,Attention,Concatenate
from tensorflow.keras import regularizers, optimizers,losses
from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives, SpecificityAtSensitivity,SensitivityAtSpecificity
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import np_utils
import numpy as np
import pandas as pd 
import matplotlib
import seaborn as sns
import sklearn
import imblearn

import matplotlib.pyplot as plt
import time
import sklearn.metrics as m

from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('mitbih/mitbih_train.csv')
test = pd.read_csv('mitbih/mitbih_test.csv')

train= np.asarray(train)
test= np.asarray(test)
x_train= train[:,0:186]
y_train= train[:,187]
y_train = to_categorical(y_train)

x_test= test[:,0:186]
y_test= test[:,187]
y_test = to_categorical(y_test)


x_train= x_train.reshape(-1,186,1)
x_test= x_test.reshape(-1,186,1)

name="CNN"
input_shape=(186,1);
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=(5), activation="relu", input_shape=(186,1)))
model.add(MaxPooling1D(pool_size=(3)))
model.add(Dropout(0.5))

model.add(Conv1D(filters=64, kernel_size=(5), activation='relu'))
model.add(MaxPooling1D(pool_size=(3)))
model.add(Dropout(0.5))

model.add(Conv1D(filters=64, kernel_size=(3), activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.8)) 
model.add(Dense(5, activation='softmax'))
print(model.summary())
plot_model(model, to_file=name+'.png',show_shapes= True , show_layer_names=True)

model.compile(loss = 'categorical_crossentropy',
                optimizer='adam',
                metrics=['acc',Recall(),Precision(),AUC(),TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])

history = model.fit(x_train, y_train,
          batch_size=500,
          epochs=100,
          validation_data=(x_test, y_test), 
          verbose=1)

model.save(name+'.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig(name+'_acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(name+'_loss.png')
plt.show()


#prd =model.predict(x)
#a= np.asarray(prd)
#print(prd)
#pd.DataFrame(a).to_csv("prediction_"+name+".csv")
pd.DataFrame.from_dict(history.history).to_csv(name+'_history.csv',index=False)

