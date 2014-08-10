.. _files:

Which files are written to disk ?
=================================

**pySkyNet** makes a system call to SkyNet, before it does so it writes
away files that SkyNet uses. Once the neural net is trained pySkyNet
reads in the results and returns them to the user. 

Files written before and after of training:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The features and targets for training:**

.. code:: python
    
    self.train_input_file  = ''.join([self.input_root,self.id,'train.txt'])

**The features and targets for validation:**

.. code:: python
    
    self.valid_input_file  = ''.join([self.input_root,self.id,'test.txt'])

**The SkyNet configuration file:**

.. code:: python

    self.network_file = ''.join([self.output_root,self.id ,'_','network.txt']) 
  
Once training has completed SkyNet writes the following files to disk:
    
**Training predictions:** 

.. code:: python
    
    self.train_pred_file = ''.join([output_root_file,'train_pred.txt'])

**Validation predictions:**

.. code:: python
    
    self.valid_pred_file = ''.join([output_root_file,'test_pred.txt'])
        
Files written after predictions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Prediction file:**

.. code:: python

    self.output_file = ''.join([self.result_root,self.id,'_predictions.txt'])

Format of prediction files:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All files that contain predictions have the same format. 
Even for a prediction where the labels or targets are not knows.
The format of the columns is follow for regression:
    
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

If the true targets/classes are not know these values are meaningless but will still be printed to file.
**pySkyNet** only returns the prediction values.
The `true_class_[n]` is printed in one-hot encoding, thus all values are zero expect for the correct class.
The sum of all values of prob_class_[n] is equal to 1.  

