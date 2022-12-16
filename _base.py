from typing import Callable
from numpy.linalg import norm
import numpy as np
from ..base import Transformer
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
from sklearn.metrics import accuracy_score


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



class Gaussian(Observable): # haide betrajje3 bs col
    '''
    Gaussians with centers are uniformally randomly distributed points from state space.
    rho: width of Gaussians
    '''
    def __init__(self, centers , rho):
        self.centers = centers
        self.rho = rho
       
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        ##psi = np.zeros((x.shape[0],1))
        ##for i in range(0,x.shape[0]): 
            ###for j in range(0,self.centers.shape[0]):
                ###gb[i][j] = np.exp(-(norm(self.centers[j]-x[i])**2)/self.rho**2)

            ##psi[i] = np.exp(-(np.linalg.norm(self.centers-x[i])**2)/self.rho**2)
        #gb = lambda x : np.exp(-(np.linalg.norm(self.centers-x)/self.rho)**2)
        ###gb/gb.sum(axis=1)[:,None]
        psi=np.array([np.exp(-(np.linalg.norm(self.centers-e)**2)/self.rho**2) for e in x]) 
        #psi = psi/np.sum(psi)
        return psi


class Rinn(Observable):  # haide betrajje3 bs col
    #y = w.x + b # Sequential as we have single input and single output otherwise we have funtion API
    #for the input layer we dont need to define act func or ker initi its just send inputs to the 1st layer
    def __init__(self, units, activations, input_dim): # for the output always better to use sigmoid 
        self.model = Sequential(
        [Input(shape = input_dim)] + [Dense(unit, activation = activation, kernel_initializer = "uniform") for unit, activation in zip(units, activations)]) # kern initi for dense layer hon unform         
               
    def _evaluate(self, x: np.ndarray, y: np.ndarray, x_test, y_test) -> np.ndarray:
        #loss = tf.keras.losses.mean_squared_error
        #optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.model.compile(optimizer="sgd", loss="mse", metrics=["mae", "acc"])
        # fit the keras model on the dataset
        self.model.fit(x, y, epochs=150, batch_size=10, verbose=0)
        # make class predictions with the model
        #predictions = (self.model.predict(x) > 0.5).astype(int)
        yhat = self.model.predict(x_test)
        # evaluate predictions
        acc = accuracy_score(y_test, yhat)
        psi= np.array([self.model.predict(x)])
        psi = np.reshape(psi, (-1,1))
        return psi
    

class Genbasis(Observable): # haide betrajje el matrix kella 
    
    def __init__(self, basis, M, *hyperParams): #fi yb3at params adma bade
        self.basis = basis
        self.M = M
        if self.basis == "Gaussianbasis":
            self.centers, self.rho = hyperParams[0], hyperParams[1] # centers yle fiha kl el centers
        
        elif self.basis == "Rinnbasis":
            self.units, self.activations = hyperParams[0], hyperParams[1]
            
    def _evaluate(self, x: np.ndarray) -> np.ndarray:        
        psis = np.zeros((x.shape[0], self.M))
        if self.basis == "Gaussianbasis":
        
            for j in range(self.M):
                psi = Gaussian(self.centers[j], self.rho)._evaluate(x)
                psis[:,j] = psi.reshape((-1,)) # kl rows w col j


        elif self.basis == "Rinnbasis":
            
            for j in range(self.M):
                psi = Rinn(self.units, self.activations, x.shape[1])._evaluate(x) #input layer 3m te5od no2ta 3nda tnen dim
                psis[:,j] = psi.reshape((-1,)) 
                
        return psis


