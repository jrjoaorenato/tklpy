import numpy as np
from tklpy.utils import kernelTypes
from tklpy.utils import kernelFinder
from scipy.sparse.linalg import eigs
from qpsolvers import solve_qp

#this implementation uses the package qpsolvers

class TKL:
    """ TKL calculations class uses numpy data structures"""
    def __init__(self, Xs, Xt, Ys, Yt, ker = kernelTypes.rbf, gamma = 1.0, eta = 2.0):
        """Constructor of the TKL class

        Arguments:
            n is the number of instances
            and f is the dimensionality of the features
            Xs {numpy Matrix [nsxf]} -- [Source domain data]
            Xt {numpy Matrix [ntxf]} -- [Target domain data]
            Ys {numpy Array with ns elements} -- [Labels of the Source Domain data]
            Yt {numpy Array with nt elements} -- [Labels of the Target Domain data, for benchmarking purposes]
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
        self.kerType = ker
        self.gamma = gamma
        self.eta = eta
    
    def findTKL(self):
        X1 = (self.Xs).transpose()
        X2 = (self.Xt).transpose()
        X = np.concatenate((X1,X2),axis=1)
        m = X1.shape[1]
        K = kernelFinder.findKernel(X, self.kerType, self.gamma)
        K = K + 1e-6 * np.eye(K.shape[1])
        Ks = K[0:m, 0:m]
        Kt = K(m:,m:)
        Kst = K(0:m,m:)
        Kat = K(:, m:)

        dim = min((Kt.shape[1] - 10), 200)
        #Lamt is the eigenvalues and Phit is the Eigenvectors
        [Lamt, Phit] = eigs((Kt + Kt.transpose())/2, dim, which='LM')
        Lamt = np.diag(Lamt,k=0)
        Phis = Kst * Phit * (np.linalg.inv(Lamt))
        Phia = Kat * Phit * (np.linalg.inv(Lamt))

        A = Phis.transpose() * Phis;
        B = Phis.transpose() * Ks * Phis;
        Q = np.multiply(A, A.transpose())
        Q = (Q + Q.transpose())/2
        r = (np.diag(B))*-1
        Anq = np.diag((-1*np.ones((dim,1)) + np.diag((self.eta * np.ones(dim-1, 1)),k=1) ))
        bnq = np.zeros((dim,1))
        lb = zeros(dim,1)
        calcLambda = solve_qp(Q,r,G=Anq,h=bnq,lb=lb,solver='quadprog')

        self.TKL = Phia * diag(calcLambda) * Phia.transpose()
        self.TKL = (self.TKL + (self.TKL).transpose())/2
        return self.TKL

    def findKernel(self):
        X1 = (self.Xs).transpose()
        X2 = (self.Xt).transpose()
        X = np.concatenate((X1,X2),axis=1)
        self.Kernel = kernelFinder.findKernel(X, self.kerType, self.gamma)
        return self.Kernel

    def returnSourceDataTKL():
        pass

    def convertDatatoTKL():
        pass
        
    def returnTargetDataTKL():
        pass
    def returnSourceDataKernel():
        pass
    def returnTargetDataKernel():
        pass
    def trainSVM():
        pass
    def testSVM():
        pass
