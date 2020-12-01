import numpy as np
from typing import Tuple
from .constitutive_base import ConstitutiveBase


class CrystalPlasticity(ConstitutiveBase):
    """
    Constitutive eq. class for crystal plasticity theory inheriting class: ConstitutiveBase
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal=metal, nitg=nitg, val=val, params=params)

    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None, plane_stress_type: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_dof = du.shape[0]

        # Velocity gradient [L]dt
        VelGrad = np.zeros((3, 3))
        VelGrad[:n_dof, :n_dof] = np.dot(du, bm.T)
        if n_dof == 2:
            VelGrad = self.calc_correction_term_plane_stress_L(VelGrad, itg, plane_stress_type)

        # Deformation rate [D]dt and spin [W]dt
        DefRate = 0.5*(VelGrad + VelGrad.T)
        Wspin = 0.5*(VelGrad - VelGrad.T)

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
