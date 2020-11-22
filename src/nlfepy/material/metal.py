from abc import abstractmethod
from .material import Material


class Metal(Material):
    """
    Metal class inheriting class: Material
    """

    def __init__(self) -> None:

        super().__init__()

        self.n_system = None
        self.base_s = None
        self.base_m = None
        self.shear_modulus = None
        self.burgers_vec = None

    @abstractmethod
    def set_crystal(self) -> None:
        pass

    @abstractmethod
    def set_shear_modulus(self) -> None:
        pass
