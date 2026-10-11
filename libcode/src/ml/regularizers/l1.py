from .base import Regularization

import numpy as np

class L1(Regularization):
    def cost(self, parameter):
        parameter = np.asarray(parameter, dtype=float)

        cost = np.sum(np.abs(parameter))

        return cost

    def gradient(self, parameter):
        parameter = np.asarray(parameter, dtype=float)

        parameter = np.sign(parameter)

        return parameter

    