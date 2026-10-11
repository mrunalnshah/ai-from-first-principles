from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def cost(self, f_wb, y):
        """
        Returns the loss
        """

    def gradient(self, f_wb, y):
        """
        Returns the gradient of the loss function
        """