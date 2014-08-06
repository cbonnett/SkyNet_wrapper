"""
Poor man's wrapper for SkyNet 
"""

# Authors: CHRISTOPHER BONNETT <c.bonnett@gmail.com>
# Licence: BSD 3 clause

import numpy as np
import write_SkyNet_files as binning
import os
from abc import ABCMeta, abstractmethod

__all__ = ["SkyNetClassifier,SkyNetRegressor"]

class SkyNet():
      
      __metaclass__ = ABCMeta
      
      @abstractmethod
      def __init__(self,
                   name,
                   input_root  ='/Users/Christopher_old/ice/data/SkyNet/train_valid/',
                   output_root ='/Users/Christopher_old/ice/data/SkyNet/network/',
                   result_root = '/Users/Christopher_old/ice/data/SkyNet/results/',
                   config_root= '/Users/Christopher_old/ice/data/SkyNet/config_files/',
                   layers=[10,10,10],
                   activation = 2220,
                   prior=1,
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
                   line_search = 2,
                   mini_batch_fraction =1.0,
                   n_jobs=1):

          self.name = name
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

          self.train_input_file  = self.input_root + self.name + 'train.txt'
          self.valid_input_file  = self.input_root + self.name + 'test.txt'

          if self.classification_network:
              binning.write_SkyNet_cla_bin(self.train_input_file,X_train,y_train)
              binning.write_SkyNet_cla_bin(self.valid_input_file,X_valid,y_valid)
              self.SkyNet_config_file =  self.config_root + self.name + '_cla.inp'
          else :
              binning.write_SkyNet_reg(self.train_input_file,X_train,y_train)
              binning.write_SkyNet_reg(self.valid_input_file,X_valid,y_valid)
              self.SkyNet_config_file =  self.config_root + self.name + '_reg.inp'

          output_root_file = self.output_root + self.name + '_'
          self.network_file = ''.join([output_root_file,'network.txt'])

          f_class = open(self.SkyNet_config_file,'w')
          print >>f_class,'#input_root'  
          print >>f_class,self.train_input_file[:-9]
          print >>f_class,'#output_root'
          print >>f_class,self.network_file[:-11]
          print >>f_class,'#classification_network'
          print >>f_class,str(self.classification_network)
          for layer in self.layers:
              print >>f_class,'#nhid'
              print >>f_class,str(layer)
          print >>f_class,'#activation'
          print >>f_class,str(self.activation)
          print >>f_class,'#prior'
          print >>f_class,str(self.prior)
          print >>f_class,'#whitenin'
          print >>f_class,str(self.whitenin)
          print >>f_class,'#whitenout'
          print >>f_class,str(self.whitenout)
          print >>f_class,'#noise_scaling'
          print >>f_class,str(self.noise_scaling)
          print >>f_class,'#set_whitened_noise'
          print >>f_class,str(self.set_whitened_noise)
          print >>f_class,'#sigma'
          print >>f_class,str(self.sigma)
          print >>f_class,'#confidence_rate'
          print >>f_class,str(self.confidence_rate)
          print >>f_class,'#confidence_rate_minimum'
          print >>f_class,str(self.confidence_rate_minimum)
          print >>f_class,'#iteration_print_frequency'
          print >>f_class,str(self.iteration_print_frequency)
          print >>f_class,'#fix_seed'
          print >>f_class,str(self.fix_seed)
          print >>f_class,'#fixed_seed'
          print >>f_class,str(self.fixed_seed)
          print >>f_class,'#calculate_evidence'
          print >>f_class,str(self.calculate_evidence)
          print >>f_class,'#resume'
          print >>f_class,str(self.resume)
          print >>f_class,'#historic_maxent'
          print >>f_class,str(self.historic_maxent)
          print >>f_class,'#recurrent'
          print >>f_class,str(self.recurrent)
          print >>f_class,'#convergence_function'
          print >>f_class,str(self.convergence_function)
          print >>f_class,'#validation_data'
          print >>f_class,str(self.validation_data)
          print >>f_class,'#verbose'
          print >>f_class,str(self.verbose)
          print >>f_class,'#pretrain'
          print >>f_class,str(self.pretrain)
          print >>f_class,'#nepoch'
          print >>f_class,str(self.nepoch)
          print >>f_class,'#max_iter'
          print >>f_class,str(self.max_iter)
          print >>f_class,'#line_search'
          print >>f_class,str(self.line_search)
          print >>f_class,'#mini-batch_fraction'
          print >>f_class,str(self.mini_batch_fraction) 
          f_class.close()

          print 'Wrote ' + self.SkyNet_config_file
          print 'Starting training'

          ### replace os.system ###
          SkyNet_run_string = ''.join(['mpirun -np ',str(self.n_jobs),' SkyNet ',self.SkyNet_config_file]) 
          os.system(SkyNet_run_string)

          ### pandas for faster read speeds ?? ####
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
    """
    --------------------------------------------------------------------------
        Data-Handling options
    --------------------------------------------------------------------------
    input_root                  root of the data files
    classification_network      0=regression, 1=classification
    mini-batch_fraction         what fraction of training data to be used?
    validation_data             is there validation data to test against?
    whitenin                    input whitening transform to use
    whitenout                   output whitening transform to use

    --------------------------------------------------------------------------
        Network and Training options
    --------------------------------------------------------------------------
    nhid                        no. of nodes in the hidden layer. For multiple hidden layers,
                                define nhid multiple times with the no. of nodes required in
                                each hidden layer in order.
    activation                  manually set activation function of layer connections
                                options are: 0=linear, 1=sigmoid, 2=tanh,
                                             3=rectified linear, 4=softsign
                                default is 1 for all hidden and 0 for output
    			    e.g. for a network with 3 layers (input, hidden & output), 10 would
    			    set sigmoid & linear activation for hidden & output layers respectively
    prior                       use prior/regularization
    noise_scaling               if noise level (standard deviation of outputs) is to be estimated
    set_whitened_noise          whether the noise is to be set on whitened data
    sigma                       initial noise level, set on (un-)whitened data
    confidence_rate             step size factor, higher values are more aggressive. default=0.1
    confidence_rate_minimum     minimum confidence rate allowed
    max_iter                    max no. of iterations allowed
    startstdev                  the standard deviation of the initial random weights
    convergence_function        function to use for convergence testing, default is 4=error squared
                                1=log-posterior, 2=log-likelihood, 3=correlation
    historic_maxent             experimental implementation of MemSys's historic maxent option
    resume                      resume from a previous job
    reset_alpha                 reset hyperparameter upon resume
    reset_sigma                 reset hyperparameters upon resume
    randomise_weights           random factor to add to saved weights upon resume
    line_search					perform line search for optimal distance
                                0 = none (default), 1 = golden section, 2 = linbcg lnsrch

    --------------------------------------------------------------------------
        Output options
    --------------------------------------------------------------------------
    output_root                 root where the resultant network will be written to
    verbose                     verbosity level of feedback sent to stdout (0=min, 3=max)
    iteration_print_frequency   stdout feedback frequency
    calculate_evidence          whether to calculate the evidence at the convergence

    --------------------------------------------------------------------------
        Autoencoder options
    --------------------------------------------------------------------------
    pretrain                    perform pre-training?
    nepoch                      number of epochs to use in pre-training (default=10)
    autoencoder                 make autoencoder network

    --------------------------------------------------------------------------
        RNN options
    --------------------------------------------------------------------------
    recurrent                   use a RNN
    norbias                     use a bias for the recurrent hidden layer connections

    --------------------------------------------------------------------------
        Debug options
    --------------------------------------------------------------------------
    fix_seed                    use a fixed seed?
    fixed_seed                  seed to use
    """
    
    def __init__(self,
                 name,
                 classification_network = 1,
                 input_root  ='/Users/Christopher_old/ice/data/SkyNet/train_valid/',
                 output_root ='/Users/Christopher_old/ice/data/SkyNet/network/',
                 result_root = '/Users/Christopher_old/ice/data/SkyNet/results/',
                 config_root= '/Users/Christopher_old/ice/data/SkyNet/config_files/',
                 layers=[10,10,10],
                 activation = 2220,
                 prior=1,
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
                 line_search = 2,
                 mini_batch_fraction =1.0,
                 n_jobs=1):
    
         self.name = name
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
        the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        
        ###########################
        #####  set file names  ####
        ###########################
        
        self.pred_input_file = self.input_root  + self.name + '_to_predict.txt'
        self.output_file     = self.result_root + self.name + '_predictions.txt'

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

        dummy_classes = np.zeros(n_samples_pred)
        binning.write_SkyNet_cla_bin(self.pred_input_file,X,dummy_classes)
        
        ###########################
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
        
        SkyNet_predictions_string = ''.join(['CalPred',' 0 1 0 ',self.network_file,' ',self.train_input_file,' ',self.pred_input_file,' ',self.output_file,' 0 0 0']) 
        os.system(SkyNet_predictions_string)

        ###########################
        ### read in prediction ####
        ###########################
        #predictions = np.loadtxt(self.output_file)#,usecols=range(self.n_features_,(self.n_classes_) + self.n_features_))
        predictions = np.loadtxt(self.output_file,usecols=[4,5])

        return predictions

class SkyNetRegressor(SkyNet):
    """
    --------------------------------------------------------------------------
        Data-Handling options
    --------------------------------------------------------------------------
    input_root                  root of the data files
    classification_network      0=regression, 1=classification
    mini-batch_fraction         what fraction of training data to be used?
    validation_data             is there validation data to test against?
    whitenin                    input whitening transform to use
    whitenout                   output whitening transform to use

    --------------------------------------------------------------------------
        Network and Training options
    --------------------------------------------------------------------------
    nhid                        no. of nodes in the hidden layer. For multiple hidden layers,
                                define nhid multiple times with the no. of nodes required in
                                each hidden layer in order.
    activation                  manually set activation function of layer connections
                                options are: 0=linear, 1=sigmoid, 2=tanh,
                                             3=rectified linear, 4=softsign
                                default is 1 for all hidden and 0 for output
    			    e.g. for a network with 3 layers (input, hidden & output), 10 would
    			    set sigmoid & linear activation for hidden & output layers respectively
    prior                       use prior/regularization
    noise_scaling               if noise level (standard deviation of outputs) is to be estimated
    set_whitened_noise          whether the noise is to be set on whitened data
    sigma                       initial noise level, set on (un-)whitened data
    confidence_rate             step size factor, higher values are more aggressive. default=0.1
    confidence_rate_minimum     minimum confidence rate allowed
    max_iter                    max no. of iterations allowed
    startstdev                  the standard deviation of the initial random weights
    convergence_function        function to use for convergence testing, default is 4=error squared
                                1=log-posterior, 2=log-likelihood, 3=correlation
    historic_maxent             experimental implementation of MemSys's historic maxent option
    resume                      resume from a previous job
    reset_alpha                 reset hyperparameter upon resume
    reset_sigma                 reset hyperparameters upon resume
    randomise_weights           random factor to add to saved weights upon resume
    line_search					perform line search for optimal distance
                                0 = none (default), 1 = golden section, 2 = linbcg lnsrch

    --------------------------------------------------------------------------
        Output options
    --------------------------------------------------------------------------
    output_root                 root where the resultant network will be written to
    verbose                     verbosity level of feedback sent to stdout (0=min, 3=max)
    iteration_print_frequency   stdout feedback frequency
    calculate_evidence          whether to calculate the evidence at the convergence

    --------------------------------------------------------------------------
        Autoencoder options
    --------------------------------------------------------------------------
    pretrain                    perform pre-training?
    nepoch                      number of epochs to use in pre-training (default=10)
    autoencoder                 make autoencoder network

    --------------------------------------------------------------------------
        RNN options
    --------------------------------------------------------------------------
    recurrent                   use a RNN
    norbias                     use a bias for the recurrent hidden layer connections

    --------------------------------------------------------------------------
        Debug options
    --------------------------------------------------------------------------
    fix_seed                    use a fixed seed?
    fixed_seed                  seed to use
    """
    def __init__(self,
                 name,
                 classification_network = 0,
                 input_root  ='/Users/Christopher_old/ice/data/SkyNet/train_valid/',
                 output_root ='/Users/Christopher_old/ice/data/SkyNet/network/',
                 result_root = '/Users/Christopher_old/ice/data/SkyNet/results/',
                 config_root= '/Users/Christopher_old/ice/data/SkyNet/config_files/',
                 layers=[10,10,10],
                 activation = 2220,
                 prior=1,
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
    
         self.name = name
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

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """   
        
        ###########################
        #####  set file names  ####
        ###########################

        self.pred_input_file = self.input_root  + self.name + '_to_predict.txt'
        self.output_file     = self.result_root + self.name + '_predictions.txt'

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

        string_for_SN2 = ''.join(['CalPred',' 0 0 0 ',self.network_file,' ',self.train_input_file,' ',self.pred_input_file,' ',self.output_file,' 0 0 0']) 

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
        os.system(string_for_SN2)

        ###########################
        ### read in prediction ####
        ###########################
        #predictions = np.loadtxt(self.output_file)#,usecols=range(self.n_features_,(self.n_classes_) + self.n_features_))
        predictions = np.loadtxt(self.output_file,usecols=[self.n_features_ +1,])

        return predictions
