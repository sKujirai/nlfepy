import numpy as np
from .metal import Metal


class HCP(Metal):
    """
    HCP crystal class inheriting class: Metal
    """

    def __init__(self) -> None:

        super().__init__()

        self.n_system = 18

        self.set_crystal()

    def set_crystal(self) -> None:
        """
        Set basis vector s & m
        """

        self.base_s = np.zeros((3, self.n_system))
        self.base_m = np.zeros((3, self.n_system))

    def set_shear_modulus(self) -> None:
        pass
