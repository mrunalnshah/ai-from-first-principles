from abc import ABC, abstractmethod

class Scaler(ABC):
    @abstractmethod
    def fit(self, X):
        """
        Learn values from X (e.g. mean, std). Must return self
        """

    @abstractmethod
    def transform(self, X):
        """
        Scale X using the values learned in fit
        """    