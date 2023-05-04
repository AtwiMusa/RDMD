from typing import Callable
from numpy.linalg import norm
import numpy as np
from ..base import Transformer
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score

#####################################



from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)



tf.keras.backend.set_floatx('float64')


tf.autograph.set_verbosity(0)

#################################


class Observable(Transformer):
    r""" An object that transforms a series of state vectors :math:`X\in\mathbb{R}^{T\times n}` into a
    series of observables :math:`\Psi(X)\in\mathbb{R}^{T\times k}`, where `n` is the dimension of each state vector
    and `k` the dimension of the observable space.
    """

    def _evaluate(self, x: np.ndarray):
        r""" Evalues the observable on input data `x`.

        Parameters
        ----------
        x : (T, n) ndarray
            Input data.

        Returns
        -------
        y : (T, m) ndarray
            Basis applied to input data.
        """
        #raise NotImplementedError()
        return 0

    def __call__(self, x: np.ndarray):
        r""" Evaluation of the observable.

        Parameters
        ----------
        x : (N, d) np.ndarray
            Evaluates the observable for N d-dimensional data points.

        Returns
        -------
        out : (N, p) np.ndarray
            Result of the evaluation for each data point.
        """
        return self._evaluate(x)

    def transform(self, data, **kwargs):
        return self(data)


class Concatenation(Observable):
    r"""Concatenation operation to evaluate :math:`(f_1 \circ f_2)(x) = f_1(f_2(x))`, where
    :math:`f_1` and :math:`f_2` are observables.

    Parameters
    ----------
    obs1 : Callable
        First observable :math:`f_1`.
    obs2 : Callable
        Second observable :math:`f_2`.
    """

    def __init__(self, obs1: Callable[[np.ndarray], np.ndarray], obs2: Callable[[np.ndarray], np.ndarray]):
        self.obs1 = obs1
        self.obs2 = obs2

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.obs1(self.obs2(x))
'''
class Gaussian(Observable): # haide betrajje3 bs col
    def __init__(self, centers , rho):
        self.centers = centers
        self.rho = rho
       
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        #psi = np.zeros((x.shape[0],1))
        #for i in range(0,x.shape[0]): 
            ###for j in range(0,self.centers.shape[0]):
                ###gb[i][j] = np.exp(-(norm(self.centers[j]-x[i])**2)/self.rho**2)

            #psi[i] = np.exp(-(np.linalg.norm(self.centers-x[i])**2)/self.rho**2)
        #gb = lambda x : np.exp(-(np.linalg.norm(self.centers-x)/self.rho)**2)
        ###gb/gb.sum(axis=1)[:,None]
        psi=np.array([np.exp(-(np.linalg.norm(self.centers-e)**2)/self.rho**2) for e in x]) 
        #psi = psi/np.sum(psi)
        return psi
'''
#################################################################
#################################################################


class Gaussian(Observable): # returns the Column
    '''
    #Gaussians with centers are uniformally randomly distributed points from state space.
    #rho: width of Gaussians
    '''
    def __init__(self, centers , rho):
        self.centers = centers
        self.rho = rho
    #_evaluate that evaluates the basis functions for an input array x.   
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        psi=np.array([np.exp(-(np.linalg.norm(self.centers-e)**2)/self.rho**2) for e in x]) 
        #psi = psi/np.sum(psi)
        return psi
       
class Rinn(Observable): #Inherits from Observable # returns column # define a class Rinn which inherit from observable  and contains single method _evaluate 
#The __init__ method initializes the class with three parameters
#for the input layer we dont need to define act func or ker initi its just send inputs to the 1st layer

    def __init__(self, units, activations, input_dim):
 #The default output layer of a Keras sequential model is a Dense layer with one neuron                  
        self.model = Sequential(
        [Input(shape = input_dim)] + [Dense(unit, activation = activation, kernel_initializer = "uniform", kernel_regularizer=tf.keras.regularizers.l2(0.01)) for unit, activation in zip(units, activations)]) 

    def n_gaus(self, x: np.ndarray) -> np.ndarray:
        
        return K.exp(-0.5 * K.square(x))
        
    def dtanh(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - tf.math.square(tf.math.tanh(x))

    
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
#Generate the predictions for the input x, the output will be an array of shape (d, m) where d is the number of input samples and m = 1 is the number of units in the output layer..
        psi= np.array([self.model.predict(x)])
        psi = np.reshape(psi, (-1,1))
        return psi # column

class Genbasis(Observable): 
    def __init__(self, basis, M, *hyperParams):  # To return as much as hyperparms for basis functions
        self.basis = basis
        self.M = M
  
        if self.basis == "Gaussianbasis":
            self.centers, self.rho = hyperParams[0], hyperParams[1] # centers that contains all the centers
        
        elif self.basis == "Rinnbasis":
            self.units, self.activations, self.train, self.batch_size, self.epochs, self.x_train, self.y_train = hyperParams[0], hyperParams[1], hyperParams[2], hyperParams[3], hyperParams[4], hyperParams[5], hyperParams[6]
            
    def _evaluate(self, x: np.ndarray) -> np.ndarray:        
        psis = np.zeros((x.shape[0], self.M))
        
        if self.basis == "Gaussianbasis":    
            for j in range(self.M):
         #evaluate the Gaussian function for a given input array x and the j-th center self.centers[j] with a width of self.rho  
                psi = Gaussian(self.centers[j], self.rho)._evaluate(x) # x.shape[0]\times 1                
                psis[:,j] = psi.reshape((-1,)) # All rows and one column j # (x.shape[0], ) assigned to jth coulmn of psis

        elif self.basis == "Rinnbasis":
            
            for j in range(self.M):
                psi_model = Rinn(self.units, self.activations, x.shape[1])#input layer takes point that has two dimesnions
                if self.train:
                    psi_model.model.compile(loss="mse", optimizer="adam", metrics=["mae"])
                    history = psi_model.model.fit(self.x_train, self.y_train, batch_size= self.batch_size, epochs= self.epochs)
                psi = psi_model._evaluate(x)    
                psis[:,j] = psi.reshape((-1,)) 
                
        return psis
        
'''
psi_model = Rinn(self.units, self.activations, x.shape[1]): This line creates an instance of the Rinn class, passing the units, activations, and the number of input dimensions as arguments. The Rinn class is a subclass of Observable and contains a Keras model that consists of a sequence of dense layers.

#######################################

psi_model.model.compile(loss="mse", optimizer="adam", metrics=["mae"]): This line compiles the Keras model inside the psi_model instance using the mean squared error (mse) as the loss function, the adam optimizer, and mean absolute error (mae) as the evaluation metric.

psi_model.model refers to the Keras Sequential model that was defined and initialized in the __init__ method of the Rinn class.

compile() is a method of the Keras model object that configures the model for training. By calling psi_model.model.compile(loss="mse", optimizer="adam", metrics=["mae"]), we are configuring the model with a mean squared error (MSE) loss function, Adam optimizer, and mean absolute error (MAE) metric.

So, psi_model.model is the Keras Sequential model object that we can use to compile and train the model.

######################

history = psi_model.model.fit(self.x_train, self.y_train, batch_size= self.batch_size, epochs= self.epochs): This line trains the Keras model inside the psi_model instance using the input x_train and output y_train data, with a given batch size and number of epochs. The fit() method returns a history object that contains the training metrics such as the loss and accuracy at each epoch.

#####################

Overall, these lines create and train a Keras model using the Rinn class with a sequence of dense layers, and compile it with a given loss function, optimizer, and evaluation metric. Then the model is trained using the input and output data for a given number of epochs, batch size, and returns the training metrics.
'''