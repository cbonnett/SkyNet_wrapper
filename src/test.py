from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from SkyNet import SkyNetRegressor
from SkyNet import SkyNetClassifier

try: 
    import seaborn as sns
except:
    pass

X,y = shuffle(load_boston().data,load_boston().target)

X_train = X[0:200]
y_train = y[0:200]

X_valid = X[200:400]
y_valid = y[200:400]

X_test =X[400:]
y_test =y[400:]

sn_reg = SkyNetRegressor(id='identification', n_jobs = 4,
                         activation = (1,2,3,0),
                         layers = (2, 2, 2),
                         max_iter = 500,
                         iteration_print_frequency = 1)

sn_reg.fit(X_train,y_train,X_valid,y_valid)


# X_class,y_class = shuffle(load_iris().data, load_iris().target)
#
# X_train = X_class[0:70]
# y_train = y_class[0:70]
#
# X_valid = X_class[70:100]
# y_valid = y_class[70:100]
#
# X_test =X_class[100:]
# y_test =y_class[100:]
#
# sn_cla = SkyNetClassifier(id = 'identification', n_jobs = 1, activation = (3,3,3,0), layers = (5,5,5), max_iter = 400,
#                           iteration_print_frequency = 1)
#
# sn_cla.fit(X_train, y_train, X_valid, y_valid)
#
# test_yhat = sn_cla.predict_proba(X_test)