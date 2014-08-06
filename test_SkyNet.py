import SkyNet as SN 
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
import os

#######################
### Classification ####
#######################

X,y = shuffle(load_iris().data,load_iris().target, random_state=0)

X_train = X[0:100]
y_train = y[0:100]

X_valid = X[100:]
y_valid = y[100:]

sn_class = SN.SkyNetClassifier('test_cla',n_jobs=8,activation=3330)
train,test = sn_class.fit(X_train,y_train,X_valid,y_valid)

pred = sn_class.predict_proba(X_valid)

####################
### regeression ####
####################

X,y = shuffle(load_boston().data,load_boston().target, random_state=0)

X_train = X[0:200]
y_train = y[0:200]

X_valid = X[200:]
y_valid = y[200:]

sn_reg = SN.SkyNetRegressor('test_reg',n_jobs=8,activation=3330)
train,test = sn_reg.fit(X_train,y_train,X_valid,y_valid)
pred = sn_reg.predict(X_valid)