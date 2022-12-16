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
        r"""
        Returns the prediction of the NN as a column vector.
        
        Parameters
        ----------
        x : (N, d) np.ndarray
            Evaluates the observable for N d-dimensional data points.

        Returns
        -------
        psi : (N, 1) np.ndarray
            Result of the prediction for each data point.
        """
        psi=np.array([np.exp(-(np.linalg.norm(self.centers-e)**2)/self.rho**2) for e in x]) 
        return psi


class Rinn(Observable): 
    def __init__(self, units, activations, input_dim): 
        # for the output it would be better to use sigmoid 
        self.model = Sequential(
        [Input(shape = input_dim)] + [Dense(unit, activation = activation, kernel_initializer = "uniform") for unit, activation in zip(units, activations)]) 
        
               
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        r"""
        Returns the prediction of the NN as a column vector.
        
        Parameters
        ----------
        x : (N, d) np.ndarray
            Evaluates the observable for N d-dimensional data points.

        Returns
        -------
        psi : (N, 1) np.ndarray
            Result of the prediction for each data point.
        """
        psi= np.array([self.model.predict(x)])
        psi = np.reshape(psi, (-1,1))
        return psi
    

class Genbasis(Observable): 
    # This would give the whole matrix 
    
    def __init__(self, basis, M, *hyperParams): 
        self.basis = basis
        self.M = M
        if self.basis == "Gaussianbasis":
            self.centers, self.rho = hyperParams[0], hyperParams[1] # centers yle fiha kl el centers
        
        elif self.basis == "Rinnbasis":
            self.units, self.activations = hyperParams[0], hyperParams[1]
            
    def _evaluate(self, x: np.ndarray) -> np.ndarray: 
        r"""
        psis: (N, M), where M is the number of basis functions
        """
        psis = np.zeros((x.shape[0], self.M))
        if self.basis == "Gaussianbasis":
        
            for j in range(self.M):
                psi = Gaussian(self.centers[j], self.rho)._evaluate(x)
                psis[:,j] = psi.reshape((-1,)) # all rows and just jth col


        elif self.basis == "Rinnbasis":
            
            for j in range(self.M):
                psi = Rinn(self.units, self.activations, x.shape[1])._evaluate(x) #input layer takes 2-dim points
                psis[:,j] = psi.reshape((-1,)) 
                
        return psis


