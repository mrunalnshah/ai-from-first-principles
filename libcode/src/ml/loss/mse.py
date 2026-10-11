from .base import Loss

import numpy as np

class MSE(Loss):
    def cost(self, f_wb, y):
        f_wb = np.asarray(f_wb, dtype=float)
        y = np.asarray(y, dtype=float)

        diff = f_wb - y
        square_diff = diff ** 2

        mse = square_diff.mean()

        return mse

    def gradient(self, f_wb, y):
        f_wb = np.asarray(f_wb, dtype=float)
        y = np.asarray(y, dtype=float)

        m = y.shape[0]

        dL_df = 2 * (f_wb - y)
        dL_df = dL_df / m

        return dL_df
