import numpy as np
from typing import Tuple
from .constitutive_base import ConstitutiveBase


class CrystalPlasticity(ConstitutiveBase):
    """
    Constitutive eq. class for crystal plasticity theory inheriting class: ConstitutiveBase
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal=metal, nitg=nitg, val=val, params=params)

    def constitutive_equation(
        self, *, du: np.ndarray, bm: np.ndarray, itg: int, plane_stress_type: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_dof = du.shape[0]

        # Velocity gradient [L]dt
        VelGrad = np.zeros((3, 3))
        VelGrad[:n_dof, :n_dof] = np.dot(du, bm.T)
        if n_dof == 2:
            VelGrad = self.calc_correction_term_plane_stress_L(
                VelGrad, itg, plane_stress_type
            )

        # Deformation rate [D]dt and spin [W]dt
        DefRate = 0.5 * (VelGrad + VelGrad.T)
        Wspin = 0.5 * (VelGrad - VelGrad.T)

        # Update thickness for 2D plane stress
        if plane_stress_type > 0:
            self._val["thickness"][itg] *= 1.0 + DefRate[2, 2]

        # Cij -> Cijkl
        Cijkl = self.get_ctensor(Cij=self._val["cmatrix"][itg])

        # Ti -> Tij
        Tij = self._val["stress"][itg, [0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(3, 3)

        # Ri -> Rij
        Rij = self._val["rvector"][itg, [0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(3, 3)

        # Jaumann rate of Cauchy stress *dt
        dTjaumann = np.tensordot(Cijkl, DefRate) - Rij

        # Update Cauchy stress
        Tij += dTjaumann + np.dot(Wspin, Tij) - np.dot(Tij, Wspin)

        # Now writing.....
        raise NotImplementedError()

        # Cijkl -> Cij
        self._val["cmatrix"][itg] = self.get_cmatrix(Cijkl=Cijkl)

        # Tij -> Ti
        self._val["stress"][itg] = Tij.flatten()[[0, 4, 8, 1, 5, 6]]

        # Rij -> Ri
        self._val["rvector"][itg] = Rij.flatten()[[0, 4, 8, 1, 5, 6]]

        return (
            self._val["cmatrix"][itg],
            self._val["rvector"][itg],
            self._val["stress"][itg],
        )
