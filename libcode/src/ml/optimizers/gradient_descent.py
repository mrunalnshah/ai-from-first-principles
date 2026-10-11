from .base import Optimizer

class GradientDescent(Optimizer):
    """
    For every parameter p, simultaneously:
        p = p - alpha * g
    
    p is weight_j or bias
    g is gradient (dJ/dp)
    """
    def step(self, params, grads):
        new_params = []

        for p, g in zip(params, grads):
            new_p = p - self.learning_rate * g
            new_params.append(new_p)

        return new_params