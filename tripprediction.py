import pandas

import matplotlib.pyplot as plt
import pandas as pd

from keras.losses import Hinge
from keras.metrics import Precision
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler

dataset = pandas.read_csv('/content/sample_data/traindata.csv')

dataset = dataset.fillna(0)

dataset = dataset.values

X = dataset[:,0:5].astype(float)
Y = dataset[:,5]

scalar = StandardScaler()
scalar.fit(X)
Xstand = scalar.transform(X)

model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=Hinge(), metrics=[Precision()],optimizer='adam')

model.fit(Xstand, Y, epochs=150, batch_size=1000, verbose=1)

dataset1 = pandas.read_csv('/content/sample_data/testdata.csv')

dataset1 = dataset1.fillna(0)

dataset1 = dataset1.values

Xnew = dataset1[:,0:5].astype(float)

Xnewstand = scalar.transform(Xnew)

ynew = model.predict_classes(Xnewstand)

file1 = open("myfile.txt","w")

for i in range(len(ynew)):
	file1.write(str(ynew[i]))
	file1.write("\n")
 
file1.close()