import numpy as np
from typing import Tuple
from abc import ABCMeta, abstractmethod


class Constitutive(metaclass=ABCMeta):
    """
    Constitutive class (base class)
    """

    def __init__(self, metal, val: dict = {}) -> None:
        self.metal = metal
        self.val = val

    @abstractmethod
    def constitutive_equation(self, *, du: np.ndarray = None, bm: np.ndarray = None, itg: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
