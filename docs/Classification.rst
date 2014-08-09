.. _classification:

An introduction to classification with pySkyNet
===============================================
Regression : Predicting a continuous value for a new example.

| In the following example we will use the ``SkyNetClassifier`` class to classify flowers in the iris dataset. 
| You will need `sklearn <http://scikit-learn.org/stable/>`_ library to load the iris dataset.  

.. code:: python

    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    from SkyNet import SkyNetRegressor

    # X are the features and y are the targets
    # shuffle returns a random permutation 
    X_class,y_class = shuffle(load_iris().data,load_iris().target)
    
We split the data into train,valid and test.
The neural net will adjust the weights solely based
on the train data. It will monitor the error on the 
valid data and stop the training once the error 
on the validation data fails to diminish
then after that we get the prediction for the
test data 
     
.. code:: python

    X_train = X[0:70]
    y_train = y[0:70]
     
    X_valid = X[70:100]
    y_valid = y[70:100]

    X_test =X[100:]
    y_test =y[100:]
    
We instantiate the neural network with 3 hidden layers with each 10 nodes ``[10,10,10]`` on the training data.
With linear rectified units as activation functions for the hidden layers and linear activation for the
outputlayer ``[3,3,3,0]`` on a single core ``n_jobs=1``. 
We use the string ``identification`` as id.

.. code:: python
    
    sn_cla = SkyNetClassifier(id='identification',n_jobs=1,activation=[3,3,3,0],layers=[10,10,10])
    
Now we perform the actual training of the neural network

.. code:: python 
    
    sn_cla.fit(X_train,y_train,X_valid,y_valid)
    
    
| Whereafter ``sn_cla.train_pred`` are the learned class probabilities  for the training set.
| Whereafter ``sn_cla.valid_pred`` are the learned class probabilities  for for the validation set.

Getting the predictions for the test set

.. code:: python

    test_yhat = sn_class.predict_proba(X_test)
    
All code combined  

.. code:: python
    
    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    from SkyNet import SkyNetRegressor

    # X are the features and y are the targets
    # shuffle returns a random permutation 
    X_class,y_class = shuffle(load_iris().data,load_iris().target)

    X_train = X[0:70]
    y_train = y[0:70]
     
    X_valid = X[70:100]
    y_valid = y[70:100]

    X_test =X[100:]
    y_test =y[100:]
    
    sn_cla = SkyNetClassifier(id='identification',n_jobs=1,activation=[3,3,3,0],layers=[10,10,10])
    
    sn_cla.fit(X_train,y_train,X_valid,y_valid)
    
    test_yhat = sn_class.predict_proba(X_test)
