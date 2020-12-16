import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Nd2(ShapeFunction):
    """
    Nd2 (2-node element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "ND"
        self._name = "ND2"
        self._n_dof = 1
        self._n_node = 2
        self._n_intgp = 2
        self._weight = np.array([1.0, 1.0])
        self._Shpfnc = np.array(
            [
                [0.788675134594813, 0.211324865405187],
                [0.211324865405187, 0.788675134594813],
            ]
        )
        self._Bmatrix_nat = np.array([[[-0.5, -0.5], [0.5, 0.5]]])
