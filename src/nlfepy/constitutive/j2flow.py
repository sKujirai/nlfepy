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

        # Deviatoric stress T'ij
        DevStress = Tij - 1.0 / 3.0 * np.trace(Tij) * np.identity(3)

        # Update equivalent strain
        self._val["eqv_strain"][itg] += (
            self._val["eqv_strain_rate"][itg] * self._params["dt"]
        )

        # Calc. equivalent stress
        self._val["eqv_stress"][itg] = np.sqrt(1.5 * np.tensordot(DevStress, DevStress))

        # Calc. flow stress
        FlowStress = self._params["ref_flow_stress"]

        # Calc. equivalent strain rate
        self._val["eqv_strain_rate"][itg] = self._params["ref_eqv_strain_rate"]
        if self._val["eqv_stress"][itg] < FlowStress:
            self._val["eqv_strain_rate"][itg] *= np.power(
                (self._val["eqv_stress"][itg] / FlowStress),
                (1.0 / self._params["strain_sensitivity"]),
            )

        # Plastic constitutive equation
        Dp = np.zeros((3, 3))
        if self._val["eqv_stress"][itg] > 0:
            Dp = (
                1.5
                * self._val["eqv_strain_rate"][itg]
                / self._val["eqv_stress"][itg]
                * DevStress
            )

        Rij = np.tensordot(Cijkl, Dp) * self._params["dt"]

        # Tij -> Ti
        self._val["stress"][itg] = Tij.flatten()[[0, 4, 8, 1, 5, 6]]

        # Rij -> Ri
        self._val["rvector"][itg] = Rij.flatten()[[0, 4, 8, 1, 5, 6]]

        return (
            self._val["cmatrix"][itg],
            self._val["rvector"][itg],
            self._val["stress"][itg],
        )
