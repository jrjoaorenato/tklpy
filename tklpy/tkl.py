import numpy as np
from tklpy.utils import kernelTypes
from tklpy.utils import kernelFinder
from scipy.sparse.linalg import eigs
from qpsolvers import solve_qp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

        self.tkl = Phia * diag(calcLambda) * Phia.transpose()
        self.tkl = (self.tkl + (self.tkl).transpose())/2
        return self.tkl

    def findKernel(self):
        X1 = (self.Xs).transpose()
        X2 = (self.Xt).transpose()
        X = np.concatenate((X1,X2),axis=1)
        self.Kernel = kernelFinder.findKernel(X, self.kerType, self.gamma)
        return self.Kernel

    def returnSourceDataTKL():
        #used to train the svm
        m = (self.Xs).transpose().shape[1]
        return self.tkl(0:m,0:m)
        
    def returnTargetDataTKL():
        #used to test the svm
        m = (self.Xs).transpose().shape[1]
        return self.tkl(m:,0:m)

    def returnSourceDataKernel():
        m = (self.Xs).transpose().shape[1]
        return self.Kernel(0:m,0:m)

    def returnTargetDataKernel():
        m = (self.Xs).transpose().shape[1]
        return self.Kernel(m:,0:m)

    def trainSVMforTKL(self):
        Xs = self.returnSourceDataTKL()
        TKsvc = SVC(kernel='precomputed')
        TKsvc.fit(Xs, self.Ys)
        self.TKsvc = TKsvc

    def testSVMforTKL(self):
        Xt = self.returnTargetDataTKL()
        TKy_pred = self.TKsvc.predict(Xt)
        acc_tkl = accuracy_score(self.Yt, TKy_pred)
        return acc_tkl
    
    def trainSVMforKernel(self):
        Xs = self.returnSourceDataKernel()
        TKsvc = SVC(kernel='precomputed')
        TKsvc.fit(Xs, self.Ys)
        self.TKsvc = TKsvc

    def testSVMforKernel(self):
        Xt = self.returnTargetDataKernel()
        TKy_pred = self.TKsvc.predict(Xt)
        acc_tkl = accuracy_score(self.Yt, TKy_pred)
        return acc_tkl

    def convertDatatoTKL():
        pass
