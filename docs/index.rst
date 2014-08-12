===================
Welcome to pySkyNet
===================

| **SkyNet** is an efficient and robust neural network training code for machine learning. It is able to train large and deep feed-forward neural networks, including autoencoders, for use in a wide range of supervised and unsupervised learning applications, such as regression, classification, density estimation, clustering and dimensionality reduction. **SkyNet** is implemented in C/C++ and fully parallelised using MPI. Currently autoencoders are not implemented in **pySkyNet**

| **pySkyNet** makes system calls to SkyNet and emulates the ``fit`` and ``predict`` user interface of `sklearn <http://scikit-learn.org/stable/>`_.
| `Here is pySkyNet github page <https://github.com/cbonnett/SkyNet_wrapper>`_. 

.. note::

   | You need to have the mpi version of **SkyNet** installed for **pySkyNet** to work!
   | Get it here http://ccpforge.cse.rl.ac.uk/gf/project/skynet/
   | SkyNet is written by Philip Graff, Farhan Feroz, Michael P. Hobson, Anthony N. Lasenby

Contents:
=========

.. raw:: html

   <div class="container-fluid">
   <div class="row">
   <div class="col-md-6">
   <h2>Getting Started:</h2>

.. toctree::
   :maxdepth: 1

   installl
   Regression
   Classification
   convergenceplots
   files
   
.. raw:: html

   </div>
   <div class="col-md-6">
   <h2>Documentation:</h2>

.. toctree::
   :maxdepth: 1
   
   SkyNet
   write_SkyNet_files

