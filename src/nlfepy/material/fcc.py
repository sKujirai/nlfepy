import numpy as np
from .metal import Metal


class FCC(Metal):
    """
    FCC crystal class inheriting class: Metal
    """

    def __init__(self) -> None:

        super().__init__()

        self.n_system = 12

        self.set_crystal()

    def set_crystal(self) -> None:
        """
        Set basis vector s & m
        """

        self.base_s = np.zeros((3, self.n_system))
        self.base_m = np.zeros((3, self.n_system))

    def set_elastic_modulus(self) -> None:
        """
        Set elastic modulus: Cmatrix [C]
        """

        C11 = self.Young / ((1. + self.Poisson) * (1. - 2.*self.Poisson)) * (1. - self.Poisson)
        C12 = self.Young / ((1. + self.Poisson) * (1. - 2.*self.Poisson)) * self.Poisson
        C44 = (C11 - C12) / 2.

        self.Cmatrix = np.array([
            [C11, C12, C12, 0., 0., 0.],
            [C12, C11, C12, 0., 0., 0.],
            [C12, C12, C11, 0., 0., 0.],
            [0., 0., 0., C44, 0., 0.],
            [0., 0., 0., 0., C44, 0.],
            [0., 0., 0., 0., 0., C44]
        ])

    def set_shear_modulus(self) -> None:
        self.shear_modulus = np.full(self.n_system, self.Young / (2. * self.Poisson))
