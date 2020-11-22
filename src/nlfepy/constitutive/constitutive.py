from abc import ABCMeta, abstractmethod


class Constitutive(metaclass=ABCMeta):
    """
    Constitutive class (base class)
    """

    def __init__(self, metal) -> None:
        self.metal = metal

    @abstractmethod
    def constitutive_equation(self) -> None:
        pass
