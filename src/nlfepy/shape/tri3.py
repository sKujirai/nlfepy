import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Tri3(ShapeFunction):
    """
    Tri3 (3-node triangular element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "TRI"
        self._name = "TRI3"
        self._n_dof = 2
        self._n_node = 3
        self._n_intgp = 1
        self._n_face = 3
        self._n_fnode = 2
        self._weight = np.array([0.5])
        self._Shpfnc = np.array(
            [[0.333333333333333], [0.333333333333333], [0.333333333333333]]
        )
        self._Bmatrix_nat = np.array([[[1.0], [0.0], [-1.0]], [[0.0], [1.0], [-1.0]]])
        self._idx_face = np.array([[1, 2], [2, 0], [0, 1]])
