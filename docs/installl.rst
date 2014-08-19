Installing pySkyNet
===================

.. note::

    | Python dependencies : numpy, pandas
    | For plotting : matplotlib, we recommend you use `seaborn <http://web.stanford.edu/~mwaskom/software/seaborn/>`_
    | `The anaconda python distribution <https://store.continuum.io/cshop/anaconda/>`_  contains all the necessary libraries for all platforms.


| As **pySkyNet** is a poor man's wrapper of SkyNet it performs system calls to **SkyNet**.
| **SkyNet** can be installed from here: http://ccpforge.cse.rl.ac.uk/gf/project/skynet/
| **pySkyNet** takes care of writing the files needed in the correct format and reading in the predictions from the files once printed by **SkyNet** and returning them to the user.

This means that
the only configuration that is needed is setting the folders
where **pySkNet** and **SkyNet** will read and write.

First clone the **pySkyNet** repository:

.. code ::

    $ git clone https://github.com/cbonnett/SkyNet_wrapper.git

Then create a system variable SKYNETPATH
to an exciting folder where you want to store **SkyNet**
training, validation, predictions, configuration and prediction files.

Bash shell:

.. code::

    $ export SKYNETPATH /path/where/you/want/to/store/skynet/data/

c-shell:

.. code::

    $ SETENV SKYNETPATH /path/where/you/want/to/store/skynet/data/

Once the $SKYNETPATH is set, in the **pySkyNet** repo you run: 

.. code:: python

    $ python install.py

This will create 4 subfolders in $SKYNETPATH:
 - $SKYNETPATH/train_valid 
    This folder will contain the training and validation files.
 - $SKYNETPATH/config_files
    This folder contains the configuration files used  by **SkyNet**
 - $SKYNETPATH/network
    This folder contains the learned weight files.
    This folder will also contain the predictions of the training and validation samples
 - $SKYNETPATH/predictions
    In this folder all the predictions are printed.
    
.. note::

    All these folders can be set when instantiating ``SkyNetRegressor``
    or ``SkyNetClassifier`` class:
    
        sn_reg = SkyNetRegressor(id='identification',input_root='/my/folder/inputs')
    
    