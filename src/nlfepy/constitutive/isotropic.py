import numpy as np
from typing import Tuple
from .constitutive import Constitutive


class Isotropic(Constitutive):
    """
    Constitutive eq. class for isotropic elastic body inheriting class: Constitutive
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal=metal, nitg=nitg, val=val, params=params)

    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        return self.val['cmatrix'][itg], self.val['rtensor'][itg], self.val['stress'][itg]
