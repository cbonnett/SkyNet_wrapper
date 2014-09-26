"""This module contains function that write 
   files to disk in the correct format as 
   expected by SkyNet.
"""
import numpy as np
import pandas as pd
import warnings

__all__ = ["write_SkyNet_cla_bin", "write_SkyNet_reg", "write_SkyNet_config_file"]


def write_SkyNet_config_file(SkyNet_config_file,
                             train_input_file,
                             network_file,
                             classification_network,
                             layers,
                             activation,
                             prior,
                             whitenin,
                             whitenout,
                             noise_scaling,
                             set_whitened_noise,
                             sigma,
                             confidence_rate,
                             confidence_rate_minimum,
                             iteration_print_frequency,
                             fix_seed,
                             fixed_seed,
                             calculate_evidence,
                             resume,
                             historic_maxent,
                             recurrent,
                             convergence_function,
                             validation_data,
                             verbose,
                             pretrain,
                             nepoch,
                             max_iter,
                             line_search,
                             mini_batch_fraction,
                             norbias,
                             reset_alpha,
                             reset_sigma,
                             randomise_weights):
    """Writes SkyNet config file to disk

        Parameters
        ----------
        All :
            see attributes of :class:`SkyNetClassfier` and :class:`SkyNetRegressor`

    """
    ### reform activation format to SkyNet standard ###
    act_temp = ''
    for act in activation:
        act_temp = ''.join([act_temp, str(act)])

    f_class = open(SkyNet_config_file, 'w')
    print >> f_class, '#input_root'
    print >> f_class, train_input_file[:-9]
    print >> f_class, '#output_root'
    print >> f_class, network_file[:-11]
    print >> f_class, '#classification_network'
    print >> f_class, str(int(classification_network))
    for layer in layers:
        print >> f_class, '#nhid'
        print >> f_class, str(layer)
    print >> f_class, '#activation'
    print >> f_class, str(act_temp)
    print >> f_class, '#prior'
    print >> f_class, str(int(prior))
    print >> f_class, '#whitenin'
    print >> f_class, str(int(whitenin))
    print >> f_class, '#whitenout'
    print >> f_class, str(int(whitenout))
    print >> f_class, '#noise_scaling'
    print >> f_class, str(int(noise_scaling))
    print >> f_class, '#set_whitened_noise'
    print >> f_class, str(int(set_whitened_noise))
    print >> f_class, '#sigma'
    print >> f_class, str(sigma)
    print >> f_class, '#confidence_rate'
    print >> f_class, str(confidence_rate)
    print >> f_class, '#confidence_rate_minimum'
    print >> f_class, str(confidence_rate_minimum)
    print >> f_class, '#iteration_print_frequency'
    print >> f_class, str(iteration_print_frequency)
    print >> f_class, '#fix_seed'
    print >> f_class, str(int(fix_seed))
    print >> f_class, '#fixed_seed'
    print >> f_class, str(fixed_seed)
    print >> f_class, '#calculate_evidence'
    print >> f_class, str(int(calculate_evidence))
    print >> f_class, '#resume'
    print >> f_class, str(int(resume))
    print >> f_class, '#norbias'
    print >> f_class, str(int(norbias))
    print >> f_class, '#reset_alpha'
    print >> f_class, str(int(reset_alpha))
    print >> f_class, '#reset_sigma'
    print >> f_class, str(int(reset_sigma))
    print >> f_class, '#randomise_weights'
    print >> f_class, str(randomise_weights)
    print >> f_class, '#historic_maxent'
    print >> f_class, str(int(historic_maxent))
    print >> f_class, '#recurrent'
    print >> f_class, str(int(recurrent))
    print >> f_class, '#convergence_function'
    print >> f_class, str(convergence_function)
    print >> f_class, '#validation_data'
    print >> f_class, str(int(validation_data))
    print >> f_class, '#verbose'
    print >> f_class, str(verbose)
    print >> f_class, '#pretrain'
    print >> f_class, str(int(pretrain))
    print >> f_class, '#nepoch'
    print >> f_class, str(nepoch)
    print >> f_class, '#max_iter'
    print >> f_class, str(max_iter)
    print >> f_class, '#line_search'
    print >> f_class, str(line_search)
    print >> f_class, '#mini-batch_fraction'
    print >> f_class, str(mini_batch_fraction)
    f_class.close()


def write_SkyNet_cla_bin(outfile, features, classes):
    """Writes a file Skynet to use with Skynet classifier.

        Parameters
        ----------
        outfile : string
            filename to be written.
        features : array of [n_samples, n_features]
            The feature array.
        classes : integer array of [n_samples]
            The classes that belong to the features.
    """
    assert len(features[:, 0]) == len(classes), 'Length of feature array not equal to length of classes array'

    outf = open(outfile, "w")
    outf.write(str(len(features[0, :])))
    outf.write("\n")
    outf.write(str(len(np.unique(classes))))
    outf.write("\n")

    if classes.min() == 1:
        classes -= 1
    elif classes.min() == 0:
        pass
    else:
        warnings.warn("minimum of classes array is not 0 or 1")
        classes -= 1

    for k in np.arange(len(classes)):
        a = ','.join([str(i) for i in features[k, :]])
        b = str(classes[k])
        outf.write(a)
        outf.write("\n")
        outf.write(b)
        outf.write("\n")
    outf.close()


def write_SkyNet_reg(outfile, features, reg):
    """Writes a file Skynet to use with Skynet regressor.

        Parameters
        ----------
        outfile : string
            filename to be written.
        features : array of [n_samples, n_features]
            The feature array.
        reg : float array of [n_samples]
            The regression values that belong to the features.
    """
    assert len(features[:, 0]) == len(reg), "Length of feature array is not equal to length of reg array"

    outf = open(outfile, "w")
    outf.write(str(len(features[0, :])))
    outf.write("\n")
    outf.write('1')
    outf.write("\n")

    for k in np.arange(len(reg)):
        a = ','.join([str(i) for i in features[k, :]])
        b = str(reg[k])
        outf.write(a)
        outf.write("\n")
        outf.write(b)
        outf.write("\n")
    outf.close()


def _parse_SkyNet_output(out,
                         iteration_frequency,
                         classification_network,
                         verbose,
                         validation_data):

    """Parse stdout from SkyNet

    Returns pandas dataframes with
    the training and validation error/corr
    per training step.
    """

    if verbose != 2:

        print 'If verbose != 2 '
        print 'pySkyNet will not return'
        print 'the error and correlation as'
        print 'a function of training step.'

        return None, None, None

    elif not validation_data:

        print 'If NOT validation_data'
        print 'pySkyNet will not return'
        print 'the error and correlation as'
        print 'a function of training step'

        return None, None, None

    else:

        train_error_array = []
        valid_error_array = []

        train_corr_array = []
        valid_corr_array = []

        train_class_array = []
        valid_class_array = []

        out = out.split()

        step_loc = [i for i,word in enumerate(out) if out[i] == 'Step']
        steps = np.arange(iteration_frequency,
                          (len(step_loc) + 1) * iteration_frequency,
                          iteration_frequency)

        for i in (step_loc):
            if classification_network:

                train_corr_array.append(float(out[i + 11]))
                train_error_array.append(float(out[i + 15]))
                train_class_array.append(float(out[i + 16][:-1]))

                valid_corr_array.append(float(out[i + 34]))
                valid_error_array.append(float(out[i + 38]))
                valid_class_array.append(float(out[i + 39][:-1]))

            if not classification_network:

                train_corr_array.append(float(out[i + 12]))
                train_error_array.append(float(out[i + 16]))

                valid_corr_array.append(float(out[i + 26]))
                valid_error_array.append(float(out[i + 30]))

        error_array = np.vstack((train_error_array, valid_error_array)).T
        corr_array = np.vstack((train_corr_array, valid_corr_array)).T

        df_error_array = pd.DataFrame(error_array, index=steps,
                                      columns=['Training error',
                                      'Validation error'])

        df_corr_array = pd.DataFrame(corr_array, index=steps,
                                     columns=['Training correlation',
                                     'Validation correlation'])

        if not classification_network:

            return df_error_array, df_corr_array, None

        if classification_network:

            class_array = np.vstack((train_class_array, valid_class_array)).T
            df_class_array = pd.DataFrame(class_array, index=steps,
                                          columns=['Training classification performance',
                                          'Validation classification performance'])
            return df_error_array, df_corr_array, df_class_array