from .base import Regularization

import numpy as np

class L2(Regularization):
    def cost(self, parameter):
        parameter = np.asarray(parameter, dtype=float)

        cost = np.sum(parameter ** 2)

        return cost

    def gradient(self, parameter):
        parameter = np.asarray(parameter, dtype=float)

        parameter = 2 * parameter

        return parameter