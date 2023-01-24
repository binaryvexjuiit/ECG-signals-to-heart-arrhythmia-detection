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

name="CNNResn"
input_shape=(186,1);
inputs = keras.Input(shape=(186,1))
cnn00 = layers.Conv1D(32, 5, activation='relu',input_shape=input_shape[0:])(inputs)
cnn00 = MaxPooling1D(pool_size=(2))(cnn00)
cnn0 = BatchNormalization(axis=-1)(cnn00)
cnn0 =  Dropout(0.2)(cnn0)

con0 = Concatenate()([cnn0,cnn00])

cnn1 = layers.Conv1D(64, 3, activation='relu')(con0)
cnn1 = MaxPooling1D(pool_size=(2))(cnn1)
cnn2 = BatchNormalization(axis=-1)(cnn1)
cnn2 =  Dropout(0.2)(cnn2)
con1 = Concatenate()([cnn1,cnn2])

cnn3 = layers.Conv1D(64, 3, activation='relu')(con1)
cnn3 = MaxPooling1D(pool_size=(2))(cnn3)
cnn4 = BatchNormalization(axis=-1)(cnn3)
cnn4 =  Dropout(0.2)(cnn4)
con2 = Concatenate()([cnn3,cnn4])

cnn5 = layers.Conv1D(64, 3, activation='relu')(con2)
cnn5 = MaxPooling1D(pool_size=(2))(cnn5)
cnn6 = BatchNormalization(axis=-1)(cnn5)
cnn6 =  Dropout(0.2)(cnn6)
con3 = layers.Add()([cnn5,cnn6])

y0= Flatten()(con3)
y0=  Dense(64,activation="relu")(y0)
y = BatchNormalization(axis=-1)(y0)
y= Dropout(0.2)(y)
con=Concatenate()([y,y0])

z=  Dense(64,activation="sigmoid")(con)
z= Dropout(0.2)(z)

w=Concatenate()([con,z])
w=  Dense(128,activation="relu")(w)
outputs= Dense(5,activation="softmax")(w)
model = keras.Model(inputs, outputs)
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

