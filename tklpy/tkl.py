from tklpy.utils import kernelTypes

class TKL:
    """ TKL calculations class """
    def __init__(self, Xs, Xt, Ys, Yt, ker = rbf, gamma = 1.0, eta = 1.1):
        """Constructore of the TKL class

        Arguments:
            n is the number of instances
            and f is the dimensionality of the features
            Xs {Matrix [nsxf]} -- [Source domain data]
            Xt {Matrix [ntxf]} -- [Target domain data]
            Ys {Array with ns elements} -- [Labels of the Source Domain data]
            Yt {Array with nt elements} -- [Labels of the Target Domain data, for benchmarking purposes]
            ker is the type of kernel present utils.kernelTypes
                rbf = 0 - is the rbf kernel
                linear = 1 - is the linear kernel
                lap = 2 - is the lap kernel - todo
            gamma reffers to the gamma proprierty of the rbf kernel
            eta - ?
                
        """
        self.Xs = Xs
        self.Xt = Xt
        self.Ys = Ys
        self.Yt = Yt
        self.ker = ker
        self.gamma = gamma
        self.eta = eta
    
    def findTKL():
        pass
    def findKernel():
        pass
    def returnTrainData():
        pass
    def returnTestData():
        pass
    def trainSVM():
        pass
    def testSVM():
        pass
