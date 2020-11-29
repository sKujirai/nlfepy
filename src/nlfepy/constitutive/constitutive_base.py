import numpy as np
from typing import Tuple
from abc import ABCMeta, abstractmethod


class ConstitutiveBase(metaclass=ABCMeta):
    """
    Constitutive class (base class)
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:

        self._metal = metal
        self._ntintgp = nitg
        self._val = val
        self._params = params

        if 'cmatrix' not in self._val:
            self._val['cmatrix'] = np.tile(self._metal.Cmatrix, (self._ntintgp, 1, 1))
        if 'rtensor' not in self._val:
            self._val['rtensor'] = np.zeros((self._ntintgp, 3, 3))
        if 'stress' not in self._val:
            self._val['stress'] = np.zeros((self._ntintgp, 3, 3))

        if 'dt' not in self._params:
            self._params['dt'] = 0.01

    @abstractmethod
    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_ctensor(self, *, Cij: np.ndarray) -> np.ndarray:
        """
        Cij [6, 6] -> Cijkl [3, 3, 3, 3]
        """

        C99 = np.concatenate([Cij, Cij[:, 3:6]], axis=1)
        C99 = np.concatenate([C99, C99[3:6, :]], axis=0)
        C99[[0, 3, 8, 6, 1, 4, 5, 7, 2], :][:, [0, 3, 8, 6, 1, 4, 5, 7, 2]]
        Cijkl = C99[[0, 3, 8, 6, 1, 4, 5, 7, 2], :][:, [0, 3, 8, 6, 1, 4, 5, 7, 2]].reshape(3, 3, 3, 3)

        return Cijkl

    def get_cmatrix(self, *, Cijkl: np.ndarray) -> np.ndarray:
        """
        Cijkl [3, 3, 3, 3] -> Cij [6, 6]
        """

        return Cijkl.reshape(9, 9)[[0, 4, 8, 1, 5, 6, 3, 7, 2], :][:, [0, 4, 8, 1, 5, 6, 3, 7, 2]][:6, :6]
