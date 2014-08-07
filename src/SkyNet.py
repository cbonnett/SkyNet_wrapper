"""
This is a (poor man's) wrapper for SkyNet neural network sofware.
You need to install SkyNet :
SkyNet is an efficient and robust neural network 
training tool for machine learning.   
"""

# Authors: CHRISTOPHER BONNETT <c.bonnett@gmail.com>
# Licence: BSD 3 clause

import numpy as np
import os
from abc import ABCMeta, abstractmethod
import subprocess
import write_SkyNet_files as binning

__all__ = ["SkyNetClassifier","SkyNetRegressor"] #,SkyNetRegressor

class SkyNet():
    """
    Skynet base class
    """
    
    __metaclass__ = ABCMeta
      
    @abstractmethod
    def __init__(self,
                 id,
                 input_root  ='/Users/Christopher_old/ice/data/SkyNet/train_valid/',
                 output_root ='/Users/Christopher_old/ice/data/SkyNet/network/',
                 result_root = '/Users/Christopher_old/ice/data/SkyNet/results/',
                 config_root= '/Users/Christopher_old/ice/data/SkyNet/config_files/',
                 layers=[10,10,10],
                 activation = [2,2,2,0],
                 prior=True,
                 confidence_rate=0.3,
                 confidence_rate_minimum=0.02,
                 iteration_print_frequency=50,
                 max_iter=2000,
                 whitenin = 1,
                 whitenout = 1,
                 noise_scaling=0,
                 set_whitened_noise=0,
                 sigma=0.035,
                 fix_seed = 0,
                 fixed_seed = 0,
                 calculate_evidence = 1,
                 resume=0,
                 historic_maxent=0,
                 recurrent = 0,
                 convergence_function = 4,
                 validation_data = 1,
                 verbose = 3,
                 pretrain = 0,
                 nepoch = 10,
                 line_search = 0,
                 mini_batch_fraction =1.0,
                 n_jobs=1):

        self.id = id
        self.input_root = input_root
        self.output_root = output_root
        self.result_root = result_root
        self.config_root = config_root
        self.layers = layers
        self.prior = prior
        self.sigma = sigma
        self.confidence_rate = confidence_rate
        self.confidence_rate_minimum = confidence_rate_minimum
        self.iteration_print_frequency = iteration_print_frequency
        self.max_iter = max_iter
        self.whitenin = whitenin
        self.whitenout = whitenout
        self.noise_scaling = noise_scaling
        self.set_whitened_noise = set_whitened_noise
        self.fix_seed = fix_seed
        self.fixed_seed = fixed_seed
        self.calculate_evidence = calculate_evidence
        self.resume =resume
        self.historic_maxent = historic_maxent
        self.recurrent = recurrent
        self.convergence_function = convergence_function
        self.validation_data = validation_data
        self.verbose = verbose
        self.pretrain = pretrain
        self.nepoch = nepoch
        self.n_jobs = n_jobs
        self.activation = activation
        self.mini_batch_fraction  = mini_batch_fraction 
        self.line_search = line_search
          
    def fit(self,X_train,y_train,X_valid,y_valid):
        """Build a Neural Net from the training set (X_train, y_train,X_valid,y_valid).
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          The training input samples.

        y : array-like, shape = [n_samples]
          The target values (class labels in classification).
        """
 
    
        if (y_train.dtype in ['int64','int32','int16','int8']) and (y_valid.dtype in ['int64','int32','int16','int8']):
            self.classification_network = 1
        else:
            self.classification_network = 0
        
        n_samples_train, self.n_features_ = X_train.shape
        n_samples_valid, valid_features   = X_valid.shape
        if self.n_features_ != valid_features:
          raise ValueError("Number of features in validation set must "
                           " match the training set. Train n_features is %s and "
                           " valid n_features is %s "
                           % (self.n_features_, valid_features))

        if self.classification_network :
          self.n_classes_ = len(np.unique(y_train))
          self.classes_   = np.unique(y_train)
          classes_valid   = np.unique(y_valid)
          if not np.array_equal(self.classes_,classes_valid):
              raise ValueError("Training and validation must have the same "
                               "number of classes. Train has  %s classes " 
                               "and valid has %s classes"
                               % (self.n_classes_), len(classes_valid))

        self.train_input_file  = self.input_root + self.id + 'train.txt'
        self.valid_input_file  = self.input_root + self.id + 'test.txt'

        if self.classification_network:
          binning.write_SkyNet_cla_bin(self.train_input_file,X_train,y_train)
          binning.write_SkyNet_cla_bin(self.valid_input_file,X_valid,y_valid)
          self.SkyNet_config_file =  self.config_root + self.id + '_cla.inp'
        else :
          binning.write_SkyNet_reg(self.train_input_file,X_train,y_train)
          binning.write_SkyNet_reg(self.valid_input_file,X_valid,y_valid)
          self.SkyNet_config_file =  self.config_root + self.id + '_reg.inp'

        output_root_file = self.output_root + self.id + '_'
        self.network_file = ''.join([output_root_file,'network.txt'])

        binning.write_SkyNet_config_file(self.SkyNet_config_file,self.train_input_file,
                                        self.network_file,self.classification_network,
                                        self.layers,self.activation,self.prior,self.whitenin,
                                        self.whitenout,self.noise_scaling,
                                        self.set_whitened_noise,self.sigma,
                                        self.confidence_rate,self.confidence_rate_minimum,
                                        self.iteration_print_frequency,self.fix_seed,
                                        self.fixed_seed,self.calculate_evidence,
                                        self.resume,self.historic_maxent,
                                        self.recurrent,self.convergence_function,
                                        self.validation_data,self.verbose,
                                        self.pretrain,self.nepoch,self.max_iter,
                                        self.line_search,self.mini_batch_fraction,
                                        )
        
        SkyNet_run_array = ''.join(['mpirun -np ',str(self.n_jobs),' SkyNet ',self.SkyNet_config_file])
        subprocess.call(SkyNet_run_array, shell=True)
        
        if self.classification_network:
          train_pred = np.loadtxt(output_root_file + 'train_pred.txt',usecols=range(self.n_classes_ + self.n_features_,(2 * self.n_classes_) + self.n_features_))
          valid_pred = np.loadtxt(output_root_file + 'test_pred.txt' ,usecols=range(self.n_classes_ + self.n_features_,(2 * self.n_classes_) + self.n_features_))
        else:
          train_pred = np.loadtxt(output_root_file + 'train_pred.txt',usecols=[self.n_features_,])
          valid_pred = np.loadtxt(output_root_file + 'test_pred.txt' ,usecols=[self.n_features_,])
        
        return train_pred,valid_pred

# =============================================================================
# Public estimators
# =============================================================================

class SkyNetClassifier(SkyNet):
    """A neural net classifier.
    
    This class calls Skynet as a classifier.

    Parameters
    ----------
    id : string, compulsory
        This is a base id used to as an identifier.
        All files written by Skynet will contain
        id in the file-name.
    input_root : string, optional (default=custom)
        The folder where SkyNet-wrapper will write and SkyNet wil look for the train 
        and validation files. This parameter is best adjusted
        in SkyNet.py
    output_root : string, optional (default=custom)
        The folder where SkyNet will write the network file
        (i.e the trained weights) 
        This parameter is best adjusted
        in SkyNet.py
    result_root : string, optional (default=custom)
        The folder where SkyNet will write prediction 
        files.This parameter is best adjusted
        in SkyNet.py
    config_root : string, optional (default=custom)
        The folder where SkyNet will write the 
        config file that it uses to train.
        This parameter is best adjusted
        in SkyNet.py
    layers : array , optional (default=[10,10,10])
        The amount of hidden layers and the amount
        of nodes per hidden layer. Default is 3 
        hidden layers with 10 nodes in each layer.
    activation : list =, optional (default=[2,2,2,0])
        Which activation function to use per layer:
        0 = linear
        1 = sigmoid
        2 = tanh
        3 = rectified linear
        4 = sofsign
        Needs to have len(layers) + 1 
        as the activation of the final
        layer needs to be set.
    prior : boolean, optional (default =True)
        Use L2 weight regularization.
        Strongly advised.
    mini-batch_fraction : float, optional(default=1.0)         
        What fraction of training data to be used in each batch
    validation_data : bool, optional (default = True)             
        Is there validation data to test against?
        Strongly advise to use to prevent overfitting 
    confidence_rate : float, optional (default=0.03)
        Initial learing rate
        Step size factor, higher values are more aggressive.
    confidence_rate_minimum : float, optional (default=0.02)     
        minimum confidence rate allowed
    iteration_print_frequency : int, optional (default=50)
        Skynet feedback frequency
    max_iter : int, optional (default=2000)
        Maxium training epochs
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for 'fit'.
    whitenin : integer, optional (default=1)
        Which input transformation to use:
        0 = none
        1 = min-max
        2 = normal.
    whitenout : integer, optional (default=1)
        Which output transformation to use:
        0 = none
        1 = min-max
        2 = normal.
    convergence_function : integer, optional (default=4) 
        Which  minimization function to use for 
        convergence testing: 
        1 = log-posterior
        2 = log-likelihood 
        3 = correlation
        4 = error squared.
    historic_maxent : bool, optional (default=False)
        Experimental implementation of MemSys's historic maxent option.
    line_search : int, optional (default = 0)
        Perform line search for optimal distance:
        0 = none 
        1 = golden section
        2 = linbcg lnsrch.
    noise_scaling : bool, optional (default = False)
        If noise level (standard deviation of outputs) is to be estimated.
    set_whitened_noise : bool, optional (default =False)          
        Whether the noise is to be set on whitened data.
    sigma : float, optional (default = 0.3)
        Initial noise level, set on (un-)whitened data.
    fix_seed : bool, optional (default =False)                     
        Use a fixed seed?
        Usefull for debugging and unit-test.
    fixed_seed : int, optional (default =0)                 
        Seed to use if fix_seed == True.
    resume : bool, optional (default = False)
        Resume from a previous job.
    reset_alpha : bool, optional (default = False)
        Reset hyperparameter upon resume.
    reset_sigma : bool, optional (default = False)
        reset hyperparameters upon resume.
    randomise_weights : float, optional (default = 0.01)
        Random factor to add to saved weights upon resume.
    verbose : int, optional (default=3)                     
        Verbosity level of feedback sent to stdout 
        by SkyNet (0=min, 3=max).
    pretrain : bool,
        Perform pre-training using
        restricted BM.
    nepoch : int, optional (default=10)                    
        Number of epochs to use in pre-training.    
    
    Attributes
    ----------
    n_features :  int
        The number of features.
    train_input_file : string
        Filename of the written training file.
    valid_input_file : string.
        Filename of the written validation file.
    SkyNet_config_file : string
        Filename of SkyNet config file.
    network_file : string
        Filename of SkyNet network file.
        This file contains the trained weights:
    
    References
    ----------
    .. [1] SKYNET: an efficient and robust neural network 
        training tool for machine learning in 
        astronomy http://arxiv.org/abs/1309.0790
    
    See also
    --------
    SkyNetRegressor
    """
 
        
    def __init__(self,
                 id,
                 classification_network = 1,
                 input_root  ='/Users/Christopher_old/ice/data/SkyNet/train_valid/',
                 output_root ='/Users/Christopher_old/ice/data/SkyNet/network/',
                 result_root = '/Users/Christopher_old/ice/data/SkyNet/results/',
                 config_root= '/Users/Christopher_old/ice/data/SkyNet/config_files/',
                 layers=[10,10,10],
                 activation = [2,2,2,0],
                 prior=True,
                 sigma=0.035,
                 confidence_rate=0.3,
                 confidence_rate_minimum=0.02,
                 iteration_print_frequency=50,
                 max_iter=2000,
                 whitenin = 1,
                 whitenout = 1,
                 noise_scaling=0,
                 set_whitened_noise=0,
                 fix_seed = 0,
                 fixed_seed = 0,
                 calculate_evidence = 1,
                 resume=0,
                 historic_maxent=0,
                 recurrent = 0,
                 convergence_function = 4,
                 validation_data = 1,
                 verbose = 3,
                 pretrain = 0,
                 nepoch = 10,
                 line_search = 0,
                 mini_batch_fraction =1.0,
                 n_jobs=1):
    
         self.id = id
         self.classification_network = classification_network
         self.input_root = input_root
         self.output_root = output_root
         self.result_root = result_root
         self.config_root = config_root
         self.layers = layers
         self.prior = prior
         self.sigma = sigma
         self.confidence_rate = confidence_rate
         self.confidence_rate_minimum = confidence_rate_minimum
         self.iteration_print_frequency = iteration_print_frequency
         self.max_iter = max_iter
         self.whitenin = whitenin
         self.whitenout = whitenout
         self.noise_scaling = noise_scaling
         self.set_whitened_noise = set_whitened_noise
         self.fix_seed = fix_seed
         self.fixed_seed = fixed_seed
         self.calculate_evidence = calculate_evidence
         self.resume =resume
         self.historic_maxent = historic_maxent
         self.recurrent = recurrent
         self.convergence_function = convergence_function
         self.validation_data = validation_data
         self.verbose = verbose
         self.pretrain = pretrain
         self.nepoch = nepoch
         self.n_jobs = n_jobs
         self.activation = activation
         self.mini_batch_fraction  = mini_batch_fraction 
         self.line_search = line_search

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        by trained neural network

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
            
        Attributes
        ----------
        output_file : String
            SkyNet writes to this file:
            result_root + self.id + _predictions.txt.

        Returns
        -------
        p : array of shape = [n_samples,n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        
        ###########################
        #####  set file names  ####
        ###########################
        
        self.pred_input_file = self.input_root  + self.id + '_to_predict.txt'
        self.output_file     = self.result_root + self.id + '_predictions.txt'

        #################################
        #####  check feature lenght  ####
        #################################

        n_samples_pred, pred_n_features_ = X.shape
        if self.n_features_ != pred_n_features_:
            raise ValueError("Number of features in prediction set must "
                             "match the training/validation set." 
                             "Training set has  %s features "
                             "Prediction set has %s features "
                             % (self.n_features_, pred_n_features_))

        dummy_classes = np.random.randint(0, high=self.n_classes_, size=n_samples_pred)
        binning.write_SkyNet_cla_bin(self.pred_input_file,X,dummy_classes)
        
        ###########################
        ### check file exitence ###
        ###########################
        
        if not os.path.isfile(self.network_file):
            raise IOError("Network file %s not found "  % (self.network_file))
        if not os.path.isfile(self.train_input_file):
            raise IOError("Input file %s not found "  % (self.train_input_file))
        if not os.path.isfile(self.pred_input_file):
            raise IOError("Prediction file %s not found " % (self.pred_input_file))

        ############################
        ### calulate predictions ###
        ############################
        
        SkyNet_predictions_string = ''.join(['CalPred',' 0 1 0 ',self.network_file,' ',self.train_input_file,' ',self.pred_input_file,' ',self.output_file,' 0 0 0'])
        subprocess.call(SkyNet_predictions_string, shell=True)

        ###########################
        ### read in prediction ####
        ###########################
        predictions = np.loadtxt(self.output_file,usecols=range(self.n_classes_ + self.n_features_,(2 * self.n_classes_) + self.n_features_))
        
        return predictions

class SkyNetRegressor(SkyNet):
    """A neural net regeressor.

    Parameters
    ----------
    id : string, compulsory
        This is a base id used to as an identifier.
        All files written by Skynet will contain
        id in the file-name.
    input_root : string, optional (default=custom)
        The folder where SkyNet-wrapper will write and SkyNet wil look for the train 
        and validation files. This parameter is best adjusted
        in SkyNet.py
    output_root : string, optional (default=custom)
        The folder where SkyNet will write the network file
        (i.e the trained weights) 
        This parameter is best adjusted
        in SkyNet.py
    result_root : string, optional (default=custom)
        The folder where SkyNet will write prediction 
        files.This parameter is best adjusted
        in SkyNet.py
    config_root : string, optional (default=custom)
        The folder where SkyNet will write the 
        config file that it uses to train.
        This parameter is best adjusted
        in SkyNet.py
    layers : array , optional (default=[10,10,10])
        The amount of hidden layers and the amount
        of nodes per hidden layer. Default is 3 
        hidden layers with 10 nodes in each layer.
    activation : list =, optional (default=[2,2,2,0])
        Which activation function to use per layer:
        0 = linear
        1 = sigmoid
        2 = tanh
        3 = rectified linear
        4 = sofsign
        Needs to have len(layers) + 1 
        as the activation of the final
        layer needs to be set.
    prior : boolean, optional (default =True)
        Use L2 weight regularization.
        Strongly advised.
    mini-batch_fraction : float, optional(default=1.0)         
        What fraction of training data to be used in each batch
    validation_data : bool, optional (default = True)             
        Is there validation data to test against?
        Strongly advise to use to prevent overfitting 
    confidence_rate : float, optional (default=0.03)
        Initial learing rate
        Step size factor, higher values are more aggressive.
    confidence_rate_minimum : float, optional (default=0.02)     
        minimum confidence rate allowed
    iteration_print_frequency : int, optional (default=50)
        Skynet feedback frequency
    max_iter : int, optional (default=2000)
        Maxium training epochs
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for 'fit'.
    whitenin : integer, optional (default=1)
        Which input transformation to use:
        0 = none
        1 = min-max
        2 = normal.
    whitenout : integer, optional (default=1)
        Which output transformation to use:
        0 = none
        1 = min-max
        2 = normal.
    convergence_function : integer, optional (default=4) 
        Which  minimization function to use for 
        convergence testing: 
        1 = log-posterior
        2 = log-likelihood 
        3 = correlation
        4 = error squared.
    historic_maxent : bool, optional (default=False)
        Experimental implementation of MemSys's historic maxent option.
    line_search : int, optional (default = 0)
        Perform line search for optimal distance:
        0 = none 
        1 = golden section
        2 = linbcg lnsrch.
    noise_scaling : bool, optional (default = False)
        If noise level (standard deviation of outputs) is to be estimated.
    set_whitened_noise : bool, optional (default =False)          
        Whether the noise is to be set on whitened data.
    sigma : float, optional (default = 0.3)
        Initial noise level, set on (un-)whitened data.
    fix_seed : bool, optional (default =False)                     
        Use a fixed seed?
        Usefull for debugging and unit-test.
    fixed_seed : int, optional (default =0)                 
        Seed to use if fix_seed == True.
    resume : bool, optional (default = False)
        Resume from a previous job.
    reset_alpha : bool, optional (default = False)
        Reset hyperparameter upon resume.
    reset_sigma : bool, optional (default = False)
        reset hyperparameters upon resume.
    randomise_weights : float, optional (default = 0.01)
        Random factor to add to saved weights upon resume.
    verbose : int, optional (default=3)                     
        Verbosity level of feedback sent to stdout 
        by SkyNet (0=min, 3=max).
    pretrain : bool,
        Perform pre-training using
        restricted BM.
    nepoch : int, optional (default=10)                    
        Number of epochs to use in pre-training.    
    
    Attributes
    ----------
    n_features :  int
        The number of features.
    train_input_file : string
        Filename of the written training file.
    valid_input_file : string.
        Filename of the written validation file.
    SkyNet_config_file : string
        Filename of SkyNet config file.
    network_file : string
        Filename of SkyNet network file.
        This file contains the trained weights:
    
    References
    ----------
    .. [1] SKYNET: an efficient and robust neural network 
        training tool for machine learning in 
        astronomy http://arxiv.org/abs/1309.0790
    
    See also
    --------
    SkyNetClassifier
    """
    
    def __init__(self,
                 id,
                 classification_network = 0,
                 input_root  ='/Users/Christopher_old/ice/data/SkyNet/train_valid/',
                 output_root ='/Users/Christopher_old/ice/data/SkyNet/network/',
                 result_root = '/Users/Christopher_old/ice/data/SkyNet/results/',
                 config_root= '/Users/Christopher_old/ice/data/SkyNet/config_files/',
                 layers=[10,10,10],
                 activation = [2,2,2,0],
                 prior=True,
                 sigma=0.035,
                 confidence_rate=0.3,
                 confidence_rate_minimum=0.02,
                 iteration_print_frequency=50,
                 max_iter=2000,
                 whitenin = 1,
                 whitenout = 1,
                 noise_scaling=0,
                 set_whitened_noise=0,
                 fix_seed = 0,
                 fixed_seed = 0,
                 calculate_evidence = 1,
                 resume=0,
                 historic_maxent=0,
                 recurrent = 0,
                 convergence_function = 4,
                 validation_data = 1,
                 verbose = 3,
                 pretrain = 0,
                 nepoch = 10,
                 line_search = 0,
                 mini_batch_fraction =1.0,
                 n_jobs=1):
    
         self.id = id
         self.classification_network = classification_network
         self.input_root = input_root
         self.output_root = output_root
         self.result_root = result_root
         self.config_root = config_root
         self.layers = layers
         self.prior = prior
         self.sigma = sigma
         self.confidence_rate = confidence_rate
         self.confidence_rate_minimum = confidence_rate_minimum
         self.iteration_print_frequency = iteration_print_frequency
         self.max_iter = max_iter
         self.whitenin = whitenin
         self.whitenout = whitenout
         self.noise_scaling = noise_scaling
         self.set_whitened_noise = set_whitened_noise
         self.fix_seed = fix_seed
         self.fixed_seed = fixed_seed
         self.calculate_evidence = calculate_evidence
         self.resume =resume
         self.historic_maxent = historic_maxent
         self.recurrent = recurrent
         self.convergence_function = convergence_function
         self.validation_data = validation_data
         self.verbose = verbose
         self.pretrain = pretrain
         self.nepoch = nepoch
         self.n_jobs = n_jobs
         self.activation = activation
         self.mini_batch_fraction  = mini_batch_fraction 
         self.line_search = line_search 
        
    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is by the trained
        neural network.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        
        Attributes
        ----------
        output_file : String
            SkyNet writes to this file:
            result_root + self.id + _predictions.txt.
        
        Returns
        -------
        y: array of shape = [n_samples]
            The predicted values.
        """   
        
        ###########################
        #####  set file names  ####
        ###########################

        self.pred_input_file = self.input_root  + self.id + '_to_predict.txt'
        self.output_file     = self.result_root + self.id + '_predictions.txt'

        ##################################
        #####  check feature lenght  #####
        ##################################

        n_samples_pred, pred_n_features_ = X.shape
        if self.n_features_ != pred_n_features_:
            raise ValueError("Number of features in prediction set must "
                             "match the training/validation set." 
                             "Training set has  %s features "
                             "Prediction set has %s features "
                             % (self.n_features_, pred_n_features_))

        dummy_classes = np.zeros(n_samples_pred)
        binning.write_SkyNet_cla_bin(self.pred_input_file,X,dummy_classes)

        SkyNet_predictions_string = ''.join(['CalPred',' 0 0 0 ',self.network_file,' ',self.train_input_file,' ',self.pred_input_file,' ',self.output_file,' 0 0 0']) 

        ############################
        ### check file exitence ###
        ###########################
        if not os.path.isfile(self.network_file):
            raise IOError("Network file %s not found "  % (self.network_file))
        if not os.path.isfile(self.train_input_file):
            raise IOError("Input file %s not found "  % (self.train_input_file))
        if not os.path.isfile(self.pred_input_file):
            raise IOError("Prediction file %s not found "  % (self.pred_input_file))

        ############################
        ### calulate predictions ###
        ############################
        subprocess.call(SkyNet_predictions_string, shell=True)
        ###########################
        ### read in prediction ####
        ###########################
        predictions = np.loadtxt(self.output_file)#,usecols=range(self.n_features_,(self.n_classes_) + self.n_features_))
        #predictions = np.loadtxt(self.output_file,usecols=[self.n_features_ +1,])

        return predictions