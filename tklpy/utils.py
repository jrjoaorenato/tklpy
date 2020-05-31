#calculate kernel with sklearn
from enum import Enum
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel

class kernelTypes(Enum):
    rbf = 0
    linear = 1
    rbfLang = 2
#    lap = 2 #todo
    
class kernelFinder:
    def findKernel(self, X, kernelType, gammaVar):
        if (kernelType == kernelTypes.rbf):
            return rbf_kernel(X.transpose(), gamma=gammaVar)
        elif (kernelType == kernelTypes.linear):
            return linear_kernel(X.transpose(), dense_output=False)
    #    elif (kernelType == kernelTypes.lap):
    #        pass
        elif (kernelType == kernelTypes.rbfLang):
            #todo fazer a implementação do kernel igual ao matlab
            pass
        else:
            raise NameError('Unsupported Kernel')

