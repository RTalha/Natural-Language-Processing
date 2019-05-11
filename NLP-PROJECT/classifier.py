import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD,RMSprop,adam
import pandas as pd

PATH="/home/rana/mypython/hadi/files"

"""
df1=pd.read_csv('/home/rana/mypython/hadi/benine_files/Patient-CT-Training-BE001.csv',header=None)
df2=pd.read_csv('/home/rana/mypython/hadi/benine_files/Patient-CT-Training-BE002.csv',header=None)
df3=pd.read_csv('/home/rana/mypython/hadi/benine_files/Patient-CT-Training-BE006.csv',header=None)
df4=pd.read_csv('/home/rana/mypython/hadi/benine_files/Patient-CT-Training-BE007.csv',header=None)
df5=pd.read_csv('/home/rana/mypython/hadi/benine_files/Patient-CT-Training-BE010.csv',header=None)
frames = [df1, df2, df3, df4,df5]
result1 = pd.concat(frames)

#result for malignant data and result1 for benine data both are concantenated to single one dataframe

framess=[result,result1]
finalresult = pd.concat(framess)

data=finalresult.values		#it is now a array of (3190,32) because we have 32 dimensional features of 3190 images
np.save("x.npy",data)		#it is now saving that array for next time usage

####CREATING LABLES

labels = np.ones((3190,),dtype='int64')

labels[0:1720]=0
labels[1721:3189]=1

np.save("y.npy",labels)		#saving the lables also
"""

data=np.load("x.npy")
labels=np.load("y.npy")

X,Y=shuffle(data,labels,random_state=2)
#spilitng the data in test and traing along lables
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=4)
#one hot encoding
y_train = np_utils.to_categorical(y_train,num_classes=2)
y_test = np_utils.to_categorical(y_test,num_classes=2)

# create model
model = Sequential()
model.add(Dense(32, input_dim=32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=21, batch_size=18)
model.summary()

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
