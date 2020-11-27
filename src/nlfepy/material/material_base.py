from abc import ABCMeta, abstractmethod


class MaterialBase(metaclass=ABCMeta):
    """
    Material class (base class)
    """

    def __init__(self) -> None:
        self.Young = None
        self.Poisson = None
        self.shear_modulus = None
        self.Cmatrix = None

    @abstractmethod
    def set_elastic_modulus(self) -> None:
        pass
