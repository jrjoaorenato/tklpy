from tklpy import utils

class TKL:
    """ TKL calculations class """
    def __init__(self, Xs, Xt, Ys, Yt, ker = rbf, gamma = 1.0, eta = 2.0):
        """Constructore of the TKL class

        Arguments:
            n is the number of instances
            and f is the dimensionality of the features
            Xs {Matrix [nsxf]} -- [Source domain data]
            Xt {Matrix [ntxf]} -- [Target domain data]
            Ys {Array with ns elements} -- [Labels of the Source Domain data]
            Yt {Array with nt elements} -- [Labels of the Target Domain data, for benchmarking purposes]
        """
        self.Xs = Xs
        self.Xt = Xt
        self.Ys = Ys
        self.Yt = Yt
    
    # def findTKL