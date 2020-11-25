import numpy as np
from typing import Tuple
from .constitutive import Constitutive


class CrystalPlasticity(Constitutive):
    """
    Constitutive eq. class for crystal plasticity theory inheriting class: Constitutive
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
        Cijkl = self.get_ctensor(Cij=self.val['cmatrix'][itg])

        # Jaumann rate of Cauchy stress *dt
        dTjaumann = np.tensordot(Cijkl, DefRate) - self.val['rtensor'][itg]

        # Update Cauchy stress
        self.val['stress'][itg] += dTjaumann + np.dot(Wspin, self.val['stress'][itg]) - np.dot(self.val['stress'][itg], Wspin)

        # Now writing.....
        raise NotImplementedError()

        # Cijkl -> Cij
        self.val['cmatrix'][itg] = self.get_cmatrix(Cijkl=Cijkl)

        return self.val['cmatrix'][itg], self.val['rtensor'][itg], self.val['stress'][itg]