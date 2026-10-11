from abc import ABC, abstractmethod

class Regularization(ABC):
    @abstractmethod
    def cost(self, parameter):
        """
        Return cost of parameter
        """

    def gradient(self, parameter):
        """
        Return gradient of parameter
        """
        