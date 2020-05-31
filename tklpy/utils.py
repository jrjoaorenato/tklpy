#calculate kernel with sklearn
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
#    lap = 2 #todo
    
class kernelFinder:
    def findKernel(self, X, kernelType, gammaVar = 1.0):
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
            return rbf_kernel(X, gamma=gammaVar)
        elif (kernelType == kernelTypes.linear):
            return linear_kernel(X, dense_output=False)
    #    elif (kernelType == kernelTypes.lap):
    #        pass
        elif (kernelType == kernelTypes.rbfLang):
            #todo fazer a implementação do kernel igual ao matlab
            pass
        else:
            raise NameError('Unsupported Kernel')

