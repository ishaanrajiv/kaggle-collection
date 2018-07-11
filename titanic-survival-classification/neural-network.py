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

def accuracy(y,y0):
    return sum([1 for x in y-y0 if x==0])/len(y)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def transform(dataframe):
    dataframe['name_dev'] = dataframe.apply(name_title, axis=1)
    dataframe['ticket_title'] = dataframe.apply(ticket_title, axis=1)
    dataframe['Cabin'] = dataframe.apply(cabin, axis=1)
    dataframe['Age'] = dataframe.apply(age, axis=1)
    dataframe['Fare'] = dataframe.apply(age, axis=1)
    # dataframe['ticket_size'] = dataframe.apply(ticket_size, axis=1)
    dataframe = dataframe.drop('Name', 1).drop('Ticket', 1)
    results = ['Survived']
    # numeric_feats = dataframe.dtypes[dataframe.dtypes != "object"].index
    label_feats = [x for x in dataframe.columns if x not in results]
    for x in label_feats:
        print(x)
        dummy = pd.get_dummies(dataframe[x]).add_suffix('_'+x)
        dataframe = dataframe.drop(x,axis=1)
        dataframe = dataframe.join(dummy)
    print("labels done")
    dataframe = dataframe.fillna(dataframe.mean())
    # dataframe[numeric_feats] = scaler.fit_transform(dataframe[numeric_feats])
    print("numerical done")
    return dataframe


def name_title(row):
    name = row['Name'].lower()
    return name.split(', ')[1].split(' ')[0]

def ticket_title(row):
    ticket = row['Ticket'].lower()
    if len(ticket.split(' '))>1:
        return ticket.split(' ')[0].replace('.','')
    else:
        return np.NaN

def ticket_size(row):
    ticket = row['Ticket'].lower()
    if len(ticket.split(' '))>1:
        return ticket.split(' ')[1]
    else:
        return ticket

def cabin(row):
    cabin = row['Cabin']
    if cabin!=np.nan:
        return str(cabin)[0].lower()
    return cabin
    
def age(row):
    age = row['Age']
    if age!=np.nan:
        return np.floor(age/5)
    return age

def fare(row):
    fare = row['Fare']
    if fare!=np.nan:
        return np.floor(fare/5)
    return fare

df = pandas.read_csv("../input/train.csv").drop('PassengerId', 1)

dataframe = transform(pandas.read_csv("../input/train.csv").drop('PassengerId', 1))
test_dataframe = transform(pandas.read_csv("../input/test.csv").drop('PassengerId', 1))

idx1 =dataframe.columns
idx2 = test_dataframe.columns
common = idx1.intersection(idx2)

dataset = dataframe.values

X = dataframe[common].values
Y = dataframe['Survived'].values

dim = len(X[0])
# test_dataset = test_dataframe.values
test_X = test_dataframe[common].values



# num_train=800
# trainx = X[0:num_train]
# trainy = Y[0:num_train]
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error
# forest_model = RandomForestRegressor(n_estimators=100,max_features=6,max_depth=6)
# forest_model.fit(trainx, trainy)

# from sklearn.svm import SVC
# clf = SVC(C=0.3,class_weight='balanced',kernel='linear')
# model = clf.fit(trainx, trainy)

# predictions = np.round(model.predict(X[0:num_train]))
# accuracy(Y[0:num_train],predictions)

# predictions = np.round(model.predict(X[num_train:]))
# accuracy(Y[num_train:],predictions)




from keras.regularizers import l2 # L2-regularisation
l2_lambda = 0.01
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import  LeakyReLU  
from keras.layers.normalization import BatchNormalization
from keras import layers

input_dim = len(X[0])
output_dim = 1

model = Sequential()
model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu',kernel_regularizer=l2(l2_lambda)))
model.add(Dropout(0.4, noise_shape=None, seed=None))
model.add(Dense(20, init='uniform', activation='relu'))
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(5, init='uniform', activation='relu'))
model.add(Dropout(0.4, noise_shape=None, seed=None))
model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))
print("defined model parameters")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("defined model compiling parameters")

model.fit(X, Y, epochs=200, batch_size=100,  verbose=2, validation_split = 0.15,shuffle=True)

test_Y = model.predict(test_X)
test_Y_final = np.array([np.round(x) for x in test_Y])


count=int(pandas.read_csv("../input/test.csv").head(1)['PassengerId'][0])
final_data = pd.DataFrame([[count+i,test_Y_final[i][0]] for i in range(len(test_Y))],columns=['PassengerId','Survived']).astype(int)
final_data.to_csv('submission.csv',index=False)
