from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    @abstractmethod
    def step(self, params, grads):
        """
        Return updated params, e.g. [weights, bias] given [dj_dw, dj_db].
        """