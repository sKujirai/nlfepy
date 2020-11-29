import numpy as np
from abc import ABCMeta, abstractmethod


class MaterialBase(metaclass=ABCMeta):
    """
    Material class (base class)
    """

    def __init__(self) -> None:
        self._Young = None
        self._Poisson = None
        self._shear_modulus = None
        self._Cmatrix = None

    @property
    def Cmatrix(self) -> np.ndarray:
        return self._Cmatrix

    @abstractmethod
    def set_elastic_modulus(self) -> None:
        pass
