import numpy as np
import copy as cp
from sklearn.base import BaseEstimator, RegressorMixin


class Model(BaseEstimator, RegressorMixin):
    model = None

    def __init__(self, input_dim,
                 hidden_dim=10,
                 output_dim=1,
                 solver='sigmoid'):
        '''
        Neural Network object
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.solver = solver
        self.U = 0
        self.V = 0
        self.S = 0
        self.H = 0
        self.alpha = 0  # for regularization
        self.W1 = None
        self.W2 = None
    # Helper function

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-0.1 * x)) - 0.5

    def predict(self, x):
        '''
        Forward pass to calculate the ouput
        '''
        x = np.matrix(x)
        if self.solver == 'sigmoid':
            hidden_activation = self.sigmoid(x @ self.W1)
        elif self.solver == 'linear':
            hidden_activation = x @ self.W1
        else:
            raise RuntimeError("solver not implemented")

        y = hidden_activation @ self.W2

        return np.array(y.flatten())[0]

    def fit(self, x, y):
        '''
        Compute W2 that lead to minimal LS
        '''

        self.W1 = np.matrix(np.random.rand(self.input_dim, self.hidden_dim))
        self.W2 = np.matrix(np.random.rand(self.hidden_dim, self.output_dim))

        X = np.matrix(x)
        Y = np.matrix(np.array(y).reshape(-1, 1))

        if self.solver == 'sigmoid':
            self.H = np.matrix(self.sigmoid(X @ self.W1))
        elif self.solver == 'linear':
            self.H = np.matrix(x @ self.W1)
        else:
            raise RuntimeError("solver not implemented")

        H = cp.deepcopy(self.H)

        self.svd(H)
        iH = np.matrix(self.V) @ np.matrix(np.diag(self.S)).I @ np.matrix(self.U).T

        self.W2 = iH * Y
        return self

    def svd(self, h):
        '''
        Compute the Singular Value Decomposition of a matrix H
        '''
        H = np.matrix(h)
        self.U, self.S, Vt = np.linalg.svd(H, full_matrices=False)
        self.V = np.matrix(Vt).T
        return np.matrix(self.U), np.matrix(self.S), np.matrix(self.V)