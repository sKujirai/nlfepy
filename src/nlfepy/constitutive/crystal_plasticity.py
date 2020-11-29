import numpy as np
from typing import Tuple
from .constitutive_base import ConstitutiveBase


class CrystalPlasticity(ConstitutiveBase):
    """
    Constitutive eq. class for crystal plasticity theory inheriting class: ConstitutiveBase
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal=metal, nitg=nitg, val=val, params=params)

    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Deformation gradient [L]dt
        DefGrad = np.zeros((3, 3))
        DefGrad[:du.shape[0], :bm.shape[0]] = np.dot(du, bm.T)

        # Deformation rate [D]dt and spin [W]dt
        DefRate = 0.5*(DefGrad + DefGrad.T)
        Wspin = 0.5*(DefGrad - DefGrad.T)

        # Cij -> Cijkl
        Cijkl = self.get_ctensor(Cij=self._val['cmatrix'][itg])

        # Jaumann rate of Cauchy stress *dt
        dTjaumann = np.tensordot(Cijkl, DefRate) - self._val['rtensor'][itg]

        # Update Cauchy stress
        self._val['stress'][itg] += dTjaumann + np.dot(Wspin, self._val['stress'][itg]) - np.dot(self._val['stress'][itg], Wspin)

        # Now writing.....
        raise NotImplementedError()

        # Cijkl -> Cij
        self._val['cmatrix'][itg] = self.get_cmatrix(Cijkl=Cijkl)

        return self._val['cmatrix'][itg], self._val['rtensor'][itg], self._val['stress'][itg]
