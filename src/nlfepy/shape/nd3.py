import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Nd3(ShapeFunction):
    """
    Nd3 (3-node element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "ND"
        self._name = "ND3"
        self._n_dof = 1
        self._n_node = 3
        self._n_intgp = 2
        self._weight = np.array([1.0, 1.0])
        self._Shpfnc = np.array(
            [
                [0.455341801261480, -0.122008467928146],
                [-0.122008467928146, 0.455341801261480],
                [0.666666666666667, 0.666666666666667],
            ]
        )
        self._Bmatrix_nat = np.array(
            [
                [
                    [-1.077350269189626, 0.077350269189626],
                    [-0.077350269189626, 1.077350269189626],
                    [1.154700538379252, -1.154700538379252],
                ]
            ]
        )
