#calculate kernel with sklearn
import numpy as np
from enum import Enum
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel

class kernelTypes(Enum):
    """Enum Class to define the types of Kernel

    Arguments:
        Enum {kernel - int} -- each number referes to
        a type of kernel
    """
    rbf = 0
    linear = 1
    rbfLang = 2
#    lap = 3 #todo
    
class kernelFinder:

    @staticmethod
    def findKernel(X, kernelType = kernelTypes.rbf, gammaVar = 1.0):
        """Returns the kernel based on the type of 
        enum kernel passed

        Arguments:
            X {Matrix} -- Data to be kernelized
            kernelType {kernelTypes enum} -- Kernel choosen
            gammaVar {float} -- gamma variable on rbf kernel

        Returns:
            [Matrix] -- Kernelized data
        """
        
        if (kernelType == kernelTypes.rbf):
            return rbf_kernel(X.transpose(), gamma=gammaVar)
        elif (kernelType == kernelTypes.linear):
            return linear_kernel(X.transpose(), dense_output=False)
    #    elif (kernelType == kernelTypes.lap):
    #        pass
        elif (kernelType == kernelTypes.rbfLang):
            #todo fazer a implementação do kernel igual ao matlab
            n1sq = np.sum(np.square(X), axis = 0)
            n1 = X.shape[1]
            D = ((np.ones((n1,1))* n1sq)).transpose() + (np.ones((n1,1))* n1sq) - (2*np.matmul(X.transpose(), X))
            gamma = gammaVar / np.mean(D)
            K = np.exp(-gamma * D)
            return K
        else:
            raise NameError('Unsupported Kernel')