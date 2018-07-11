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

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

def transform(dataframe):
    if "SalePrice" in dataframe.dtypes.index:
        dataframe["SalePrice"] = np.log1p(dataframe["SalePrice"])
    numeric_feats = dataframe.dtypes[dataframe.dtypes != "object"].index
    label_feats = dataframe.dtypes[dataframe.dtypes == "object"].index
    for x in label_feats:
        print(x)
        dummy = pd.get_dummies(dataframe[x]).add_suffix('_'+x)
        dataframe = dataframe.drop(x,axis=1)
        dataframe = dataframe.join(dummy)
    print("labels done")
    dataframe = dataframe.fillna(dataframe.mean())
    for x in numeric_feats:
         if x!="SalePrice":
            print(x)
            dataframe[x] = (dataframe[x] - np.average(dataframe[x]))/(np.max(dataframe[x]) - np.min(dataframe[x]))
    print("numerical done")
    return dataframe


dataframe = transform(pandas.read_csv("../input/train.csv").drop('Id', 1))
test_dataframe = transform(pandas.read_csv("../input/test.csv").drop('Id', 1))

idx1 =dataframe.columns
idx2 = test_dataframe.columns
common = idx1.intersection(idx2)

dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataframe.drop('SalePrice',axis=1)[common].values
Y = dataframe['SalePrice'].values

dim = len(X[0])
# test_dataset = test_dataframe.values
test_X = test_dataframe[common].values


############### NEURAL NETWORK ###########################
# from keras.regularizers import l2 # L2-regularisation
# l2_lambda = 0.01
# from keras.layers import Dense, Activation, Dropout
# from keras.layers.advanced_activations import  LeakyReLU  
# from keras.layers.normalization import BatchNormalization

# model = Sequential()

# model.add(Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu',kernel_regularizer=l2(l2_lambda)))
# model.add(Dropout(0.4, noise_shape=None, seed=None))
# model.add(Dense(100, init='uniform', activation='relu'))
# model.add(Dropout(0.3, noise_shape=None, seed=None))
# model.add(Dense(1,activation='linear')) 

# model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')# Fit the model
# history = model.fit(X, Y, epochs=1000,  verbose=2, validation_split = 0.15,shuffle=True)

##############################################################


############ Bayesian Ridge ####################
from sklearn import linear_model
reg = linear_model.BayesianRidge()
model = reg.fit(X,Y) 
model.predict(X)
###############################




predictions = np.expm1(model.predict(test_X))
count=int(pandas.read_csv("../input/test.csv").head(1)['Id'][0])
arr=[]
for y in predictions:
    # print(str(count)+","+str(y[0]))
    arr.append([count,y])
    count+=1

final_data = pd.DataFrame(arr,columns=['Id','SalePrice'])
final_data.to_csv('submission.csv',index=False)

