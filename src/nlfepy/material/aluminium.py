import numpy as np
from .fcc import FCC


class Aluminium(FCC):
    """
    Aluminium class inheriting class: FCC
    """

    def __init__(self) -> None:

        super().__init__()

        self.Young = 90.3e9
        self.Poisson = 0.35
        self.burgers_vec = np.full(self.n_system, 2.56e-10)

        self.set_elastic_modulus()
        self.set_shear_modulus()
