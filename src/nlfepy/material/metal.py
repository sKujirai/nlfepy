import numpy as np
from abc import abstractmethod
from .material_base import MaterialBase


class Metal(MaterialBase):
    """
    Metal class inheriting class: MaterialBase
    """

    def __init__(self) -> None:

        super().__init__()

        self._n_system: int
        self._base_s: np.ndarray
        self._base_m: np.ndarray
        self._burgers_vec: np.ndarray

    @abstractmethod
    def set_crystal(self) -> None:
        pass

    @abstractmethod
    def set_shear_modulus(self) -> None:
        pass
