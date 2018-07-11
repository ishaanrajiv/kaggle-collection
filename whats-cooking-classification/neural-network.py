# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

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


import json
train_data = json.load(open("../input/train.json"))
test_data = json.load(open("../input/test.json"))


ingredients = {}
cuisines = {}
for num in train_data:
    for ingredient in num.get('ingredients',[]):
        ingredients[ingredient] = ingredients.get(ingredient,0)+1
    cuisines[num.get('cuisine')] = cuisines.get(num.get('cuisine'),0)+1


ingredients_dict = {v: k for k, v in dict(enumerate(ingredients.copy())).items()}

cuisines_dict = {v: k for k, v in dict(enumerate(cuisines.copy())).items()}
reverse_cuisines_dict = {k: v for k, v in  dict(enumerate(cuisines.copy())).items()}

print(len(ingredients_dict),len(train_data))
train_arr=np.zeros([len(train_data),len(ingredients_dict)+1])
for i in range(len(train_data)):
    train_arr[i][-1]=cuisines_dict[train_data[i]['cuisine']]
    for ing in train_data[i]['ingredients']:
        if ing in ingredients_dict:
            idx = ingredients_dict[ing]
            train_arr[i][idx]=1

print(len(ingredients_dict),len(test_data))
test_arr=np.zeros([len(test_data),len(ingredients_dict)])
for i in range(len(test_data)):
    # test_arr[i][-1]=cuisines_dict[test_data[i]['cuisine']]
    for ing in test_data[i]['ingredients']:
        if ing in ingredients_dict:
            idx = ingredients_dict[ing]
            test_arr[i][idx]=1


#########################

traindf = pd.DataFrame(train_arr)
testdf = pd.DataFrame(test_arr)

traindf = traindf.rename(columns={ traindf.columns[-1]: "cuisine" })

train_X = traindf.drop('cuisine',axis=1).values
train_Y_df = traindf['cuisine']
train_Y_df = pd.get_dummies(train_Y_df).add_suffix('_cuisine')
train_Y = train_Y_df.values

test_X = testdf.values

######################

num_train = int(0.9*len(train_X))

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
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(250, init='uniform', activation='relu'))
# model.add(Dropout(0.5, noise_shape=None, seed=None))
# model.add(Dense(100, init='uniform', activation='relu'))
model.add(Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
print("defined model parameters")


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("defined model compiling parameters")

model.fit(train_X[0:num_train], train_Y[0:num_train], epochs=50, batch_size=1000,  verbose=1, validation_data = (train_X[num_train:], train_Y[num_train:]))

print("model trained")
# model.predict(test_X)

test_Y = model.predict(test_X)
test_Y_final = np.array([reverse_cuisines_dict[np.argmax(x)] for x in test_Y])


final_data = pd.DataFrame([[test_data[i]['id'],test_Y_final[i]] for i in range(len(test_data))],columns=['id','cuisine'])
final_data.to_csv('submission.csv',index=False)
