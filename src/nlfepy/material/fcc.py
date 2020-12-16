import numpy as np
from .metal import Metal


class FCC(Metal):
    """
    FCC crystal class inheriting class: Metal
    """

    def __init__(self) -> None:

        super().__init__()

        self._n_system = 12

        self.set_crystal()

    def set_crystal(self) -> None:
        """
        Set basis vector s & m
        """

        self._base_s = np.zeros((3, self._n_system))
        self._base_m = np.zeros((3, self._n_system))

    def set_elastic_modulus(self) -> None:
        """
        Set elastic modulus: Cmatrix [C]
        """

        C11 = (
            self._Young
            / ((1.0 + self._Poisson) * (1.0 - 2.0 * self._Poisson))
            * (1.0 - self._Poisson)
        )
        C12 = (
            self._Young
            / ((1.0 + self._Poisson) * (1.0 - 2.0 * self._Poisson))
            * self._Poisson
        )
        C44 = (C11 - C12) / 2.0

        self._Cmatrix = np.array(
            [
                [C11, C12, C12, 0.0, 0.0, 0.0],
                [C12, C11, C12, 0.0, 0.0, 0.0],
                [C12, C12, C11, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, C44, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, C44],
            ]
        )

    def set_shear_modulus(self) -> None:
        self._shear_modulus = np.full(
            self._n_system, self._Young / (2.0 * self._Poisson)
        )
