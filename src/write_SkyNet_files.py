import numpy as np

__all__ =["write_SkyNet_cla_bin","write_SkyNet_reg"] # public interface

def write_SkyNet_cla_bin(outfile,data,classes):
    '''
    writes away a file suitable for SkyNet.
    outifle = filename to be written
    data    = array with the features 
    classes =  binned spectroscopic redsifts
    '''
    assert len(data[:,0]) == len(classes), 'Length of data array not equal to length of z array'

    outf = open(outfile,"w")
    outf.write(str(len(data[0,:])))
    outf.write("\n")
    outf.write(str(len(np.unique(classes))))
    outf.write("\n")

    if classes.min() == 1:
        classes = classes - 1
    elif classes.min() == 0:
        pass
    else :
        raise Exception("min classes is not 0 or 1")

    for k in np.arange(len(classes)):
        a = ','.join([str(i) for i in data[k,:]])
        b = str(classes[k])
        outf.write(a)
        outf.write("\n")
        outf.write(b)
        outf.write("\n")
    outf.close()

def write_SkyNet_reg(outfile,data,z):
    '''
    writes away a file suitable for SkyNet.
    outifle = filename to be written
    data    = array with the features 
    z       = regression
    '''
    assert len(data[:,0]) == len(z), 'Length of data array not equal to length of z array'

    outf = open(outfile,"w")
    outf.write(str(len(data[0,:])))
    outf.write("\n")
    outf.write('1')
    outf.write("\n")

    for k in np.arange(len(z)):
        a = ','.join([str(i) for i in data[k,:]])
        b = str(z[k])
        outf.write(a)
        outf.write("\n")
        outf.write(b)
        outf.write("\n")
    outf.close()