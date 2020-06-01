# tklpy
<!-- WIP: Need to better explain how TKL works -->

Python implementation of the Transfer Kernel Learning method developed by Long et. al. (2015). It's available as package and based on the original implementation of the TKL method from the origial authors, this implementation can be found [here](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transfer-kernel-learning-tkde15.zip).

After using this package consider citing the original author paper with:

```
M. Long, J. Wang, J. Sun and P. S. Yu, "Domain Invariant Transfer Kernel Learning," in IEEE Transactions on Knowledge and Data Engineering, vol. 27, no. 6, pp. 1519-1532, 1 June 2015, doi: 10.1109/TKDE.2014.2373376.
```

# Table of Contents
* [tklpy](#tklpy)
* [Table of Contents](#table-of-contents)
* [Installation](#installation)
    * [Method 1 - PIP](#method-1---pip)
    * [Method 2 - Manual Installation](#method-2---manual-installation)
* [Usage](#usage)
* [Example](#example)

## Installation

### Method 1 - PIP
Just run:
```pip install tklpy``` 

### Method 2 - Manual Installation

First, clone the directory, by running:

```git clone https://github.com/jrjoaorenato/tklpy.git```

The package requires the following dependencies:
- numpy==1.18.1
- qpsolvers==1.3
- quadprog==0.1.7
- scikit-learn==0.22.1
- scipy==1.4.1

You can install those dependencies by running the following command in the package directory:
``` pip install -r requirements.txt ```

After that you can run the following command in the package directory to install the package:
``` pip install -e . ```

## Usage

You can import the package by using:
``` from tklpy.tkl import TKL ```

You will also most likely need the enum list of the Kernel Types, which can be found importing:
``` from tklpy.tkl import kernelTypes ```

The types of kernel that can be found are:

0. ```rbf``` - RBF Kernel using the sklearn implementation
1. ```linear``` - Linear Kernel using the sklearn implementation
2. ```rbfLang``` - RBF Kernel using the original authors implementation and gamma calculation

To access the Kernel types you can use: `kernelTypes.<your Kernel>`.

After properly importing the package you can create an instance of the TKL class, with the following attributes:
```TKL(Xs, Xt, Ys, Yt, ker = kernelTypes.rbf, gamma = 1.0, eta = 1.1)

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
```

After creating your instance `tk` of the TKL class, you can find the Domain Invariant Kernel by running `tk.findTKL()` or you can find the regular Kernel by running `tk.findKernel()`. You can also use the methods `tk.returnSourceDataTKL()`, `tk.returnTargetDataTKL()`, `tk.returnSourceDataKernel()` and `tk.returnTargetDataKernel()` to return the desired domain Data projected into the desired Kernel.

You can also train and test a simple SVM implemented on sklearn after finding the kernels, by running `tk.trainSVMforTKL()` or `tk.trainSVMforKernel()` and `tk.testSVMforTKL()` or `tk.testSVMforKernel()`. The `trainSVM` functions will train an SVM with the respective kernel projected source data and the `testSVM` functions will test the svm with the respective kernel projected target data.

## Example
```python
from tklpy.tkl import TKL
from tklpy.utils import kernelTypes
import numpy as np
from scipy.io import loadmat

D = loadmat(<some .mat dataset location>)
Xs = D['Xs']
Xt = D['Xt']
Ys = D['Ys']
Yt = D['Yt']

tk = TKL(Xs,Xt,Ys,Yt,kernelTypes.rbfLang, 1.0, 1.1)
tk.findTKL()
tk.findKernel()
tk.trainSVMforTKL()
tk.trainSVMforKernel()
print("TKL")
tk.testSVMforTKL()
print("Kernel")
tk.testSVMforKernel()
```
