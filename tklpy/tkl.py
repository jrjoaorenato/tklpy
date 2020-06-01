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
    def __init__(self, Xs, Xt, Ys, Yt, ker = kernelTypes.rbf, gamma = 1.0, eta = 1.1):
        """Constructor of the TKL class

        Arguments:
            n is the number of instances
            and f is the dimensionality of the features
            Xs {numpy Matrix [NS x F]} -- [Source domain data]
            Xt {numpy Matrix [NT x F]} -- [Target domain data]
            Ys {numpy Array with ns elements} -- [Labels of the Source Domain data]
            Yt {numpy Array with nt elements} -- [Labels of the Target Domain data, for benchmarking purposes]
            ker is the type of kernel present utils.kernelTypes
                rbf = 0 - is the rbf kernel
                linear = 1 - is the linear kernel
                lap = 2 - is the lap kernel - todo
            gamma reffers to the gamma proprierty of the rbf kernel
            eta is the eigenspectrum damping factor
                
        """
        self.Xs = Xs
        self.Xt = Xt
        self.Ys = Ys
        self.Yt = Yt
        self.kerType = ker
        self.gamma = gamma
        self.eta = eta
    
    def findTKL(self):
        """ This function is used to find the TKL Kernel 
        that can be used to feed an SVM

        Returns:
            Matrix -- The Domain Invariant Kernel Matrix
        """
        X1 = (self.Xs).transpose()
        X2 = (self.Xt).transpose()
        X = np.concatenate((X1,X2),axis=1)
        m = X1.shape[1]
        K = kernelFinder.findKernel(X, self.kerType, self.gamma)
        K = K + 1e-6 * np.eye(K.shape[1])
        Ks = K[0:m, 0:m]
        Kt = K[m:,m:]
        Kst = K[0:m,m:]
        Kat = K[:, m:]

        dim = min((Kt.shape[1] - 10), 200)
        #Lamt is the eigenvalues and Phit is the Eigenvectors
        [Lamt, Phit] = eigs((Kt + Kt.transpose())/2, dim, which='LM')
        Lamt = np.real(Lamt)
        Phit = np.real(Phit)
        Lamt = np.diag(Lamt)
        Phis = np.matmul(np.matmul(Kst, Phit),(np.linalg.inv(Lamt)))
        Phia = np.matmul(np.matmul(Kat, Phit),(np.linalg.inv(Lamt)))

        #todo check this converter later
        self.phixlamx = np.matmul(Phit, np.linalg.inv(Lamt))

        A = np.matmul(Phis.transpose(), Phis)
        B = np.matmul(np.matmul(Phis.transpose(), Ks), Phis)
        Q = np.multiply(A.transpose(), A)
        Q = (Q + Q.transpose())/2
        r = (np.diag(B))*-1
        Anq = np.diag(-np.ones((dim))) + np.diag((self.eta * np.ones((dim-1))), 1)
        bnq = np.zeros((dim,1))
        lb = np.zeros((dim,1))
        calcLambda = solve_qp(P=Q,q=r,G=Anq,h=bnq.ravel(),lb=lb.ravel(),solver='quadprog')

        #todo check this converter later 2
        self.lamphis = np.matmul(np.diag(calcLambda), (Phis.transpose()))

        self.tkl = np.matmul(np.matmul(Phia, np.diag(calcLambda)), Phia.transpose())
        self.tkl = (self.tkl + (self.tkl).transpose())/2
        return self.tkl

    def findKernel(self):
        """Find the Kernel with the data and parameters
        specified in the constructor for TKL comparison

        Returns:
            Matrix -- Kernel specified in the constructor
        """
        X1 = (self.Xs).transpose()
        X2 = (self.Xt).transpose()
        X = np.concatenate((X1,X2),axis=1)
        self.Kernel = kernelFinder.findKernel(X, self.kerType, self.gamma)
        return self.Kernel

    def returnSourceDataTKL(self):
        """Returns the  instances of the Source Data
        projected in the Domain Invariant Kernel
        Matrix for training the algorithm

        Returns:
            Matrix -- Source Data on the Domain Invariant Kernel
        """
        #used to train the svm
        m = (self.Xs).transpose().shape[1]
        return self.tkl[0:m,0:m]
        
    def returnTargetDataTKL(self):
        """Returns the  instances of the Target Data
        projected in the Domain Invariant Kernel
        Matrix for testing the algorithm

        Returns:
            Matrix -- Target Data on the Domain Invariant Kernel
        """
        #used to test the svm
        m = (self.Xs).transpose().shape[1]
        return self.tkl[m:,0:m]

    def returnSourceDataKernel(self):
        """Returns the  instances of the Source Data
        projected in the original Kernel training
        the algorithm

        Returns:
            Matrix -- Source Data on the Kernel
        """
        m = (self.Xs).transpose().shape[1]
        return self.Kernel[0:m,0:m]

    def returnTargetDataKernel(self):
        """Returns the  instances of the Target Data
        projected in the original Kernel training
        the algorithm

        In this case for fair comparison, the data
        extracted is of the Target Data within the
        Source Data Kernel for Kernel transference
        analysis

        Returns:
            Matrix -- Target Data on the Kernel
        """
        m = (self.Xs).transpose().shape[1]
        return self.Kernel[m:,0:m]

    def trainSVMforTKL(self):
        """Trains a Support Vector Machine based on 
        libsvm with the data obtained from the method
        'returnSourceDataTKL()'
        """
        Xs = self.returnSourceDataTKL()
        TKsvc = SVC(kernel='precomputed', decision_function_shape='ovo')
        TKsvc.fit(Xs, self.Ys.ravel())
        self.TKsvc = TKsvc

    def testSVMforTKL(self):
        """Tests the Support Vector Machine obtained
        from the method 'trainSVMforTKL()' the data
        obtained from the method 'returnTargetDataTKL()'
        """
        Xt = self.returnTargetDataTKL()
        TKy_pred = self.TKsvc.predict(Xt)
        acc_tkl = accuracy_score(self.Yt.ravel(), TKy_pred)
        return acc_tkl
    
    def trainSVMforKernel(self):
        """Trains a Support Vector Machine based on 
        libsvm with the data obtained from the method
        'returnSourceDataKernel()'
        """
        Xs = self.returnSourceDataKernel()
        TKsvc = SVC(kernel='precomputed', decision_function_shape='ovo')
        TKsvc.fit(Xs, self.Ys.ravel())
        self.TKsvc = TKsvc

    def testSVMforKernel(self):
        """Tests the Support Vector Machine obtained
        from the method 'trainSVMforKernel()' the data
        obtained from the method 'returnTargetDataKernel()'
        """
        Xt = self.returnTargetDataKernel()
        TKy_pred = self.TKsvc.predict(Xt)
        acc_tkl = accuracy_score(self.Yt.ravel(), TKy_pred)
        return acc_tkl

    def convertDatatoTKL(self, X):
        """Receives a data input X and projects it 
        into the Domain Invariant Kernel Matrix,
        using this case we can use different instances
        of data for testing

        Arguments:
            X {Data Matrix [Nd x f]} -- Data to be projected
            into the Domain Invariant Kernel, in this case,
            Nd is the number of samples and f is the 
            dimensionality of the features
        """
        X1 = X.transpose()
        X2 = (self.Xt).transpose()
        Xk = np.concatenate((X1,X2),axis=1)
        Kx0x = kernelFinder.findKernel(X.transpose(), self.kerType, self.gamma)

        Phix0 = Kx0x * self.phixlamx
        return Phix0 * self.lamphis