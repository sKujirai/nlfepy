import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Tet4(ShapeFunction):
    """
    Tet4 (4-node tetrahedral element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "TET"
        self._name = "TET4"
        self._n_dof = 3
        self._n_node = 4
        self._n_intgp = 1
        self._n_face = 4
        self._n_fnode = 3
        self._weight = np.array([0.166666666666667])
        self._Shpfnc = np.array([[0.25], [0.25], [0.25], [0.25]])
        self._Bmatrix_nat = np.array(
            [
                [[-1.0], [1.0], [0.0], [0.0]],
                [[-1.0], [0.0], [1.0], [0.0]],
                [[-1.0], [0.0], [0.0], [1.0]],
            ]
        )
        self._idx_face = np.array([[0, 1, 2], [2, 3, 0], [1, 0, 3], [3, 2, 1]])
