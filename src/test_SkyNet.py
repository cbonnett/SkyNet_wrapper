from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
import os
import numpy as np
import SkyNet as SN 

#######################
### Classification ####
#######################

X_class,y_class = shuffle(load_iris().data,load_iris().target, random_state=0)

X_train_class = X_class[0:70]
y_train_class = y_class[0:70]

X_valid_class = X_class[70:100]
y_valid_class = y_class[70:100]

X_test_class = X_class[100:]
y_test_class = y_class[100:]

sn_class = SN.SkyNetClassifier('unit_test_cla',n_jobs=8,activation=[3,3,3,0],max_iter=50,fix_seed=1,fixed_seed=0)
train_class,test_class = sn_class.fit(X_train_class,y_train_class,X_valid_class,y_valid_class)

pred_class = sn_class.predict_proba(X_test_class)

####################
### regeression ####
####################

X,y = shuffle(load_boston().data,load_boston().target, random_state=0)

X_train = X[0:200]
y_train = y[0:200]

X_valid = X[200:]
y_valid = y[200:]

sn_reg = SN.SkyNetRegressor('unit_test_reg',n_jobs=8,activation=[3,3,3,0],max_iter=50,fix_seed=1,fixed_seed=0)
train,test = sn_reg.fit(X_train,y_train,X_valid,y_valid)
pred_reg = sn_reg.predict(X_valid)