#calculate kernel with sklearn
from enum import Enum
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel

class kernelTypes(Enum):
    rbf = 0
    linear = 1
#    lap = 2 #todo
    
class kernelFinder:
    def findKernel(X1, X2, kernelType, gamma):
        if (kernelType == kernelTypes.rbf):
            return rbf_kernel(X1, X2, gamma)
        elif (kernelType == kernelTypes.linear):
            return linear_kernel(X1, X2)
    #    elif (kernelType == kernelTypes.lap):
    #        pass
        else:
            raise NameError('Unsupported Kernel')

