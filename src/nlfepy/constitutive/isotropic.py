import numpy as np
from typing import Tuple
from .constitutive import Constitutive


class Isotropic(Constitutive):
    """
    Constitutive eq. class for isotropic elastic body inheriting class: Constitutive
    """

    def __init__(self, metal, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal, val)

        self.params = params

    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        Cmatrix = self.metal.Cmatrix
        Rmatrix = np.zeros((3, 3))
        Tmatrix = np.zeros((3, 3))

        return Cmatrix, Rmatrix, Tmatrix
