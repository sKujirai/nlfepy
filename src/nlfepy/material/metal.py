from abc import abstractmethod
from .material_base import MaterialBase


class Metal(MaterialBase):
    """
    Metal class inheriting class: MaterialBase
    """

    def __init__(self) -> None:

        super().__init__()

        self._n_system = None
        self._base_s = None
        self._base_m = None
        self._burgers_vec = None

    @abstractmethod
    def set_crystal(self) -> None:
        pass

    @abstractmethod
    def set_shear_modulus(self) -> None:
        pass
