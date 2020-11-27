import numpy as np
from typing import Tuple
from .constitutive_base import ConstitutiveBase


class Isotropic(ConstitutiveBase):
    """
    Constitutive eq. class for isotropic elastic body inheriting class: ConstitutiveBase
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal=metal, nitg=nitg, val=val, params=params)

    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        return self.val['cmatrix'][itg], self.val['rtensor'][itg], self.val['stress'][itg]
