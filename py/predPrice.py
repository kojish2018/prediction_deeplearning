import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import plot_model

from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal
from keras import optimizers

df1=csv.reader(open("file.csv","r")) #Training Data
data1 = [ v for v in df1]
mat = np.array(data1)
mat2 = mat[1:]
x_data = mat2[:, 1:].astype(np.float)
print('x_data.shape=', x_data.shape)

df2 = csv.reader(open("file_label.csv", "r")) #Label Data
data2 = [ v for v in df2]
mat3 = np.array(data2)
mat4 = mat3[1:]     
t_data = mat4[:, 1:].astype(np.float)
print('t_data.shape=', t_data.shape)

maxlen = 20             # The number of days for inputs
n_in = x_data.shape[1]   # The number of rows for the training data
n_out = t_data.shape[1]  # The number of rows for the label data
len_seq = x_data.shape[0] - maxlen + 1
data = []
target = []
for i in range(0, len_seq):
  data.append(x_data[i:i+maxlen, :])
  target.append(t_data[i+maxlen-1, :])

x = np.array(data).reshape(len(data), maxlen, n_in)
t = np.array(target).reshape(len(data), n_out)

print(x.shape, t.shape)

n_train = int(len(data)*0.8)              
x_train,x_test = np.vsplit(x, [n_train])  
t_train,t_test = np.vsplit(t, [n_train]) 
print(x_train.shape, x_test.shape, t_train.shape, t_test.shape)

class Prediction :
  def __init__(self, maxlen, n_hidden, n_in, n_out):
    self.maxlen = maxlen
    self.n_hidden = n_hidden
    self.n_in = n_in
    self.n_out = n_out

  def create_model(self):
    model = Sequential()
    model.add(LSTM(self.n_hidden, batch_input_shape = (None, self.maxlen, self.n_in),
             kernel_initializer = glorot_uniform(seed=20161011),  
             recurrent_initializer = orthogonal(gain=1.0, seed=20161011),
             dropout = 0.5, 
             recurrent_dropout = 0.5))
    model.add(Dropout(0.5))
    model.add(Dense(self.n_out, 
            kernel_initializer = glorot_uniform(seed=20161011)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer = "Adam" , metrics = ['categorical_accuracy'])
    return model

  # Training
  def train(self, x_train, t_train, batch_size, epochs) :
    early_stopping = EarlyStopping(patience=0, verbose=1)
    model = self.create_model()
    model.fit(x_train, t_train, batch_size = batch_size, epochs = epochs, verbose = 1,
          shuffle = True, callbacks = [early_stopping], validation_split = 0.2)
    return model


n_hidden = 80     
epochs = 15      
batch_size = 10   

# Model
prediction = Prediction(maxlen, n_hidden, n_in, n_out)
# Training
model = prediction.train(x_train, t_train, batch_size, epochs)
# Test
score = model.evaluate(x_test, t_test, batch_size = batch_size, verbose = 1)
print("score:", score)

# Accuracy and Semi-Accuracy
preds = model.predict(x_test)
correct = 0
semi_correct = 0
for i in range(len(preds)):
  pred = np.argmax(preds[i,:])
  tar = np.argmax(t_test[i,:])
  if pred == tar :
    correct += 1
  else :
    if pred+tar == 1 or pred+tar == 5 :
      semi_correct += 1

print("Accuracy:", 1.0 * correct / len(preds))
print("Semi-Accuracy(Up or Down):", 1.0 * (correct+semi_correct) / len(preds))

preds = model.predict(x_test)  #Detailed Prediction Accuracy
correct_veryrise = 0
correct_rise = 0
correct_fall = 0
correct_veryfall = 0
for i in range(len(preds)):
  pred = np.argmax(preds[i,:])
  tar = np.argmax(t_test[i,:])
  if tar==0 and pred == tar:
    correct_veryrise += 1
  elif tar==1 and pred == tar:
    correct_rise += 1
  elif tar==2 and pred == tar:
    correct_fall += 1
  elif tar==3 and pred == tar:
    correct_veryfall += 1

SUM=np.sum(t_test, axis=0)    
print("High Rise Accuracy:",correct_veryrise/SUM[0])  
print("Rise Accuracy:",correct_rise/SUM[1])  
print("Fall Accuracy:",correct_fall/SUM[2])
print("Low Fall Accuracy:",correct_veryfall/SUM[3])  

preds = model.predict(x_test)  #Detailed Prediction Semi-Accuracy
semi_correct_rise = 0
semi_correct_fall = 0

for i in range(len(preds)):
  pred = np.argmax(preds[i,:])
  tar = np.argmax(t_test[i,:])
  if pred+tar == 1 or ( pred==tar and (tar==1 or 0)) : 
    semi_correct_rise+=1
  else:
    if pred+tar == 5 or ( pred==tar and (tar==3 or 2)):
        semi_correct_fall+=1
        

SUM=np.sum(t_test, axis=0)    
print("Rise Semi-Accuracy:",semi_correct_rise/(SUM[0]+SUM[1])) 
print("Fall Semi-Accuracy:",semi_correct_fall/(SUM[2]+SUM[3]))