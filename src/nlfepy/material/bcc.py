import numpy as np
from .metal import Metal


class BCC(Metal):
    """
    BCC crystal class inheriting class: Metal
    """

    def __init__(self) -> None:

        super().__init__()

        self._n_system = 48

        self.set_crystal()

    def set_crystal(self) -> None:
        """
        Set basis vector s & m
        """

        self._base_s = np.zeros((3, self._n_system))
        self._base_m = np.zeros((3, self._n_system))

    def set_shear_modulus(self) -> None:
        pass
