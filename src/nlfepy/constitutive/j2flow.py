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

        if 'eqv_strain' not in self.val:
            self.val['eqv_strain'] = np.zeros(self.ntintgp)
        if 'eqv_strain_rate' not in self.val:
            self.val['eqv_strain_rate'] = np.zeros(self.ntintgp)
        if 'eqv_stress' not in self.val:
            self.val['eqv_stress'] = np.zeros(self.ntintgp)

        if 'ref_stress' not in self.params:
            self.params['ref_flow_stress'] = 1.e7
        if 'ref_eqv_strain_rate' not in self.params:
            self.params['ref_eqv_strain_rate'] = 1.
        if 'strain_sensitivity' not in self.params:
            self.params['strain_sensitivity'] = 0.02

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

        # Deviatoric stress T'ij
        DevStress = self.val['stress'][itg] - 1./3.*np.trace(self.val['stress'][itg])*np.identity(3)

        # Update equivalent strain
        self.val['eqv_strain'][itg] += self.val['eqv_strain_rate'][itg]*self.params['dt']

        # Calc. equivalent stress
        self.val['eqv_stress'][itg] = np.sqrt(1.5 * np.tensordot(DevStress, DevStress))

        # Calc. flow stress
        FlowStress = self.params['ref_flow_stress']

        # Calc. equivalent strain rate
        self.val['eqv_strain_rate'][itg] = self.params['ref_eqv_strain_rate']
        if self.val['eqv_stress'][itg] < FlowStress:
            self.val['eqv_strain_rate'][itg] *= np.power((self.val['eqv_stress'][itg] / FlowStress), (1. / self.params['strain_sensitivity']))

        # Plastic constitutive equation
        Dp = np.zeros((3, 3))
        if self.val['eqv_stress'][itg] > 0:
            Dp = 1.5 * self.val['eqv_strain_rate'][itg] / self.val['eqv_stress'][itg] * DevStress

        self.val['rtensor'][itg] = np.tensordot(Cijkl, Dp)*self.params['dt']

        return self.val['cmatrix'][itg], self.val['rtensor'][itg], self.val['stress'][itg]
