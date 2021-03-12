import sys
import numpy as np
from typing import Tuple
from .constitutive_base import ConstitutiveBase


class J2flow(ConstitutiveBase):
    """
    Constitutive eq. class for J2 flow theory inheriting class: ConstitutiveBase
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:
        super().__init__(metal=metal, nitg=nitg, val=val, params=params)

        if "eqv_strain" not in self._val:
            self._val["eqv_strain"] = np.zeros(self._ntintgp)
        if "eqv_strain_rate" not in self._val:
            self._val["eqv_strain_rate"] = np.zeros(self._ntintgp)
        if "eqv_stress" not in self._val:
            self._val["eqv_stress"] = np.zeros(self._ntintgp)

        if "ref_stress" not in self._params:
            self._params["ref_flow_stress"] = 1.0e7
        if "ref_eqv_strain_rate" not in self._params:
            self._params["ref_eqv_strain_rate"] = 1.0
        if "strain_sensitivity" not in self._params:
            self._params["strain_sensitivity"] = 0.02

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

        # Deviatoric stress T'ij
        DevStress = Tij - 1.0 / 3.0 * np.einsum(
            "b, bij -> bij",
            np.trace(Tij, axis1=1, axis2=2),
            np.tile(np.identity(3), (n_intgp, 1, 1)),
        )

        # Update equivalent strain
        self._val["eqv_strain"][itg] += (
            self._val["eqv_strain_rate"][itg] * self._params["dt"]
        )

        # Calc. equivalent stress
        self._val["eqv_stress"][itg] = np.sqrt(
            1.5 * np.einsum("bij, bij -> b", DevStress, DevStress)
        )

        # Calc. flow stress
        FlowStress = self._params["ref_flow_stress"]

        # Calc. equivalent strain rate
        self._val["eqv_strain_rate"][itg] = self._params["ref_eqv_strain_rate"]

        idx_l = np.where(self._val["eqv_stress"][itg] < FlowStress)
        idx_g = itg[idx_l]
        self._val["eqv_strain_rate"][idx_g] *= np.power(
            (self._val["eqv_stress"][idx_g] / FlowStress),
            (1.0 / self._params["strain_sensitivity"]),
        )

        # Plastic constitutive equation
        Dp = np.zeros((n_intgp, 3, 3))
        idx = np.where(self._val["eqv_stress"][itg] > 0.0)
        Dp[idx] = 1.5 * np.einsum(
            "b, bij -> bij",
            self._val["eqv_strain_rate"][idx] / self._val["eqv_stress"][idx],
            DevStress[idx],
        )

        Rij = np.einsum("bijkl, bkl -> bij", Cijkl, Dp) * self._params["dt"]

        # Tij -> Ti
        self._val["stress"][itg] = Tij.reshape(-1, 9)[:, [0, 4, 8, 1, 5, 6]]

        # Rij -> Ri
        self._val["rvector"][itg] = Rij.reshape(-1, 9)[:, [0, 4, 8, 1, 5, 6]]

        return (
            self._val["cmatrix"][itg],
            self._val["rvector"][itg],
            self._val["stress"][itg],
        )
