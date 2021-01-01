import numpy as np
from abc import ABCMeta, abstractmethod


class MaterialBase(metaclass=ABCMeta):
    """
    Material class (base class)
    """

    def __init__(self) -> None:
        self._Young: float
        self._Poisson: float
        self._shear_modulus: np.ndarray
        self._Cmatrix: np.ndarray

    @property
    def Cmatrix(self) -> np.ndarray:
        return self._Cmatrix

    @abstractmethod
    def set_elastic_modulus(self) -> None:
        pass
