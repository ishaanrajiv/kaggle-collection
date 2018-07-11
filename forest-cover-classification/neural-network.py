import os
print(os.listdir("../input"))

import numpy
import pandas
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import numpy as np
from scipy.stats import skew

# Any results you write to the current directory are saved as output.

traindf = pd.read_csv("../input/train.csv").drop('Id', 1)
testdf = pd.read_csv("../input/test.csv").drop('Id', 1)

# traindf=
traindf_normalized = (traindf-traindf.min())/(traindf.max()-traindf.min()).fillna(0)
testdf_normalized = (testdf-testdf.min())/(testdf.max()-testdf.min()).fillna(0)


train_X = traindf_normalized.fillna(0).drop('Cover_Type',axis=1).values
train_Y_df = traindf['Cover_Type']
train_Y_df = pd.get_dummies(train_Y_df)
train_Y = train_Y_df.values

test_X = testdf_normalized.fillna(0).values

# num_train = int(0.9*len(train_X))

from keras.regularizers import l2 # L2-regularisation
l2_lambda = 0.01
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import  LeakyReLU  
from keras.layers.normalization import BatchNormalization
from keras import layers

input_dim = len(train_X[0])
output_dim = len(train_Y[0])

model = Sequential()
model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu',kernel_regularizer=l2(l2_lambda)))
model.add(Dropout(0.4, noise_shape=None, seed=None))
model.add(Dense(30, init='uniform', activation='relu'))
model.add(Dropout(0.3, noise_shape=None, seed=None))
# model.add(Dense(20, init='uniform', activation='relu'))
# model.add(Dropout(0.4, noise_shape=None, seed=None))
model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
print("defined model parameters")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("defined model compiling parameters")

model.fit(train_X, train_Y, epochs=500, batch_size=100,  verbose=2, validation_split = 0.2)

print("model trained")

test_Y = model.predict(test_X)
test_Y_final = np.array([np.argmax(x) for x in test_Y])

print(test_Y[0:10])
print(test_Y_final[0:10])

idx = pd.read_csv("../input/test.csv").values[0][0]
final_data = pd.DataFrame([[idx+i,int(test_Y_final[i])+1] for i in range(len(test_Y))],columns=['Id','Cover_Type'])
final_data.to_csv('submission.csv',index=False)
