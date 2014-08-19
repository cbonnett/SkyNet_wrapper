.. _regression:

An introduction to regression with pySkyNet
===========================================

Regression : Predicting a continuous value for a new example.

| In the following example we will use the ``SkyNetRegressor`` class to estimate Boston housing prices. 
| You will need `sklearn <http://scikit-learn.org/stable/>`_ library to load the Boston dataset.  

.. code:: python

    from sklearn.datasets import load_boston
    from sklearn.utils import shuffle
    from SkyNet import SkyNetRegressor

    # X are the features and y are the targets
    # shuffle returns a random permutation 
    X,y = shuffle(load_boston().data, load_boston().target)
    
We split the data into train,valid and test.
The neural net will adjust the weights solely based
on the train data. It will monitor the error on the 
validation data and stop the training once the error 
on the validation data fails to diminish.
After that we calculate the predictions for the
test data using the trained neural network.

.. code:: python

    X_train = X[0:200]
    y_train = y[0:200]
     
    X_valid = X[200:400]
    y_valid = y[200:400]

    X_test =X[400:]
    y_test =y[400:]
    
We instantiate the neural network with 3 hidden layers with each 10 nodes ``(10, 10, 10)``.
With linear rectified units as activation functions for the hidden layers and linear activation for the
outputlayer ``(3, 3, 3, 0)`` on a single core ``n_jobs=1``.
We use the string ``'identification'`` as id.

.. code:: python
    
    sn_reg = SkyNetRegressor(id = 'identification_reg', n_jobs = 1, activation = (3, 3, 3, 0), layers = (0, 10, 10), max_iter = 200)
    
Now we perform the actual training of the neural network

.. code:: python 
    
    sn_reg.fit(X_train,y_train,X_valid,y_valid)
    
| Whereafter ``sn_reg.train_pred`` are the learned regression values for the training set.
| Whereafter ``sn_reg.train_pred`` are the learned regression values for the validation set.

Getting the predictions for the test

.. code:: python

    test_yhat = sn_reg.predict(X_test)
    
Compare the predictions 

.. code:: python

   >>> print 'Mean error squared training    = ',((y_train - sn_reg.train_pred) ** 2).sum()/float(len(y_train))
   >>> print 'Mean error squared validation  = ',((y_valid - sn_reg.valid_pred) ** 2).sum()/float(len(y_valid))
   >>> print 'Mean error squared test        = ',((y_test - test_yhat)   ** 2).sum()/float(len(y_test))
   Mean error squared training    =  10.9789472372
   Mean error squared validation  =  15.7155137255
   Mean error squared test        =  9.75942040179
   
All code combined  

.. code:: python

   from sklearn.datasets import load_boston
   from sklearn.utils import shuffle
   from SkyNet import SkyNetRegressor

   # X are the features and y are the targets
   # shuffle returns a random permutation 
   X,y = shuffle(load_boston().data,load_boston().target)

   X_train = X[0:200]
   y_train = y[0:200]
    
   X_valid = X[200:400]
   y_valid = y[200:400]

   X_test =X[400:]
   y_test =y[400:]
   
   sn_reg = SkyNetRegressor(id = 'identification_reg', n_jobs = 1, activation=(3, 3, 3, 0), layers = (10, 10, 10), max_iter=200)
   
   sn_reg.fit(X_train,y_train,X_valid,y_valid)
   
   test_yhat = sn_reg.predict(X_test)
   
   print 'Mean error squared training    = ',((y_train - sn_reg.train_pred) ** 2).sum()/float(len(y_train))
   print 'Mean error squared validation  = ',((y_valid - sn_reg.valid_pred) ** 2).sum()/float(len(y_valid))
   print 'Mean error squared test        = ',((y_test - test_yhat)   ** 2).sum()/float(len(y_test))