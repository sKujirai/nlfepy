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
        self,
        *,
        du: np.ndarray,
        bm: np.ndarray,
        itg: np.ndarray,
        plane_stress_type: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_intgp, n_dof, _ = bm.shape

        # Velocity gradient [L]dt
        VelGrad = np.zeros((n_intgp, 3, 3))
        VelGrad[:, :n_dof, :n_dof] = np.matmul(du, bm.transpose(0, 2, 1))
        if n_dof == 2:
            VelGrad = self.calc_correction_term_plane_stress_L(
                VelGrad, itg, plane_stress_type
            )

        # Deformation rate [D]dt and spin [W]dt
        DefRate = 0.5 * (VelGrad + VelGrad.transpose(0, 2, 1))
        Wspin = 0.5 * (VelGrad - VelGrad.transpose(0, 2, 1))

        # Update thickness for 2D plane stress
        if plane_stress_type > 0:
            self._val["thickness"][itg] *= 1.0 + DefRate[:, 2, 2]

        # Cij -> Cijkl
        Cijkl = self.get_ctensor(Cij=self._val["cmatrix"][itg])

        # Ti -> Tij
        Tij = self._val["stress"][itg][:, [0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(-1, 3, 3)

        # Ri -> Rij
        Rij = self._val["rvector"][itg][:, [0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(
            -1, 3, 3
        )

        # Jaumann rate of Cauchy stress *dt
        dTjaumann = np.einsum("bijkl, bkl -> bij", Cijkl, DefRate) - Rij

        # Update Cauchy stress
        Tij += dTjaumann + np.matmul(Wspin, Tij) - np.matmul(Tij, Wspin)

        # Now writing.....
        raise NotImplementedError()

        # Cijkl -> Cij
        self._val["cmatrix"][itg] = self.get_cmatrix(Cijkl=Cijkl)

        # Tij -> Ti
        self._val["stress"][itg] = Tij.reshape(-1, 9)[:, [0, 4, 8, 1, 5, 6]]

        # Rij -> Ri
        self._val["rvector"][itg] = Rij.reshape(-1, 9)[:, [0, 4, 8, 1, 5, 6]]

        return (
            self._val["cmatrix"][itg],
            self._val["rvector"][itg],
            self._val["stress"][itg],
        )
