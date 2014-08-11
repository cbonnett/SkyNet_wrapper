.. _files:

Which files are written to disk ?
=================================

**pySkyNet** makes a system call to **SkyNet**, before it does so it writes
away files that **SkyNet** uses. Once the neural net is trained **pySkyNet**
reads in the results and returns them to the user. 

| See examples below.

Files written before training:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The features and targets for training:**

.. code:: python
    
    self.train_input_file  = ''.join([self.input_root, self.id, '_train.txt'])

**The features and targets for validation:**

.. code:: python
    
    self.valid_input_file  = ''.join([self.input_root, self.id,'_test.txt'])

**The SkyNet configuration file:**

.. code:: python

    self.SkyNet_config_file = ''.join([self.config_root, self.id, '_reg.inp']) #Regression
    self.SkyNet_config_file = ''.join([self.config_root, self.id, '_cla.inp']) #Classification

Files written after training:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once training has completed SkyNet writes the following files to disk:

**Training predictions:** 

.. code:: python
    
    self.train_pred_file = ''.join([output_root_file, '_train_pred.txt'])

**Validation predictions:**

.. code:: python
    
    self.valid_pred_file = ''.join([output_root_file, '_test_pred.txt'])
        
**Learned weights file:**

.. code:: python

    self.network_file = ''.join([output_root_file, 'network.txt'])

Files written after predicting:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Prediction file:**

.. code:: python

    self.output_file = ''.join([self.result_root,self.id, '_predictions.txt'])

Format of prediction files:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All files that contain predictions have the same format. 
regardless if labels or targets are not known or not.

|The format of the columns is follows for regression predictions:
    
.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 20px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
    .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 20px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
    .tg .tg-7khl{font-size:15px}
    </style>
    <table class="tg">
      <tr>
        <th class="tg-7khl">feature_1</th>
        <th class="tg-031e">feature_2</th>
        <th class="tg-031e">...</th>
        <th class="tg-031e">feauture_n</th>
        <th class="tg-031e">true_target</th>
        <th class="tg-031e">pred_taget</th>
      </tr>
    </table>

|

For Classification it is as follows:

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 20px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
    .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 20px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
    .tg .tg-7khl{font-size:15px}
    </style>
    <table class="tg">
      <tr>
        <th class="tg-7khl">feature_1</th>
        <th class="tg-031e">feature_2</th>
        <th class="tg-031e">...</th>
        <th class="tg-031e">feauture_n</th>
        <th class="tg-031e">true_class_1</th>
        <th class="tg-031e">...</th>
        <th class="tg-031e">true_class_n</th>
        <th class="tg-031e">prob_class_1</th>
        <th class="tg-031e">...</th>
        <th class="tg-031e">prob_class_n</th>
      </tr>
    </table>

|

If the true targets/classes are not know the 'true' values are meaningless, but they will still be printed to file.
**pySkyNet** only returns the prediction values.
The `true_class_[n]` is printed in one-hot encoding, thus all values are zero expect for the correct class.
The sum of all values of prob_class_[n] is equal to 1.

Examples:
~~~~~~~~~

.. code::

    sn_reg = SkyNetRegressor(id='identification', n_jobs=1, activation=[3,3,3,0], layers=[10,10,10], max_iter=200)
    sn_reg.fit(X_train,y_train,X_valid,y_valid)
    test_yhat = sn_reg.predict(X_test)

After which:

.. code::

    >>> print sn_reg.train_input_file
    $SKYNETPATH/train_valid/identification_train.txt
    >>> print sn_reg.test_input_file
    $SKYNETPATH/train_valid/identification_test.txt
    >>> print sn_reg.SkyNet_config_file
    $SKYNETPATH/config_files/identification_reg.inp
    >>> print sn_reg.train_pred_file
    $SKYNETPATH/network/identification_train_pred.txt
    >>> print sn_reg.valid_pred_file
    $SKYNETPATH/network/identification_test_pred.txt
    >>> print sn_reg.network_file
    $SKYNETPATH/network/identification_network.txt
    >>> print sn_reg.output_file
    $SKYNETPATH/predictions/identification_predictions.txt
    
    

