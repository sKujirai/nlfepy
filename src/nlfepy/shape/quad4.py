import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Quad4(ShapeFunction):
    """
    Quad4 (4-node quadrilateral element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "QUAD"
        self._name = "QUAD4"
        self._n_dof = 2
        self._n_node = 4
        self._n_intgp = 4
        self._n_face = 4
        self._n_fnode = 2
        self._weight = np.array([1.0, 1.0, 1.0, 1.0])
        self._Shpfnc = np.array(
            [
                [0.622008, 0.166667, 0.166667, 0.0446582],
                [0.166667, 0.622008, 0.0446582, 0.166667],
                [0.0446582, 0.166667, 0.166667, 0.622008],
                [0.166667, 0.0446582, 0.622008, 0.166667],
            ]
        )
        self._Bmatrix_nat = np.array(
            [
                [
                    [-0.394338, -0.394338, -0.105662, -0.105662],
                    [0.394338, 0.394338, 0.105662, 0.105662],
                    [0.105662, 0.105662, 0.394338, 0.394338],
                    [-0.105662, -0.105662, -0.394338, -0.394338],
                ],
                [
                    [-0.394338, -0.105662, -0.394338, -0.105662],
                    [-0.105662, -0.394338, -0.105662, -0.394338],
                    [0.105662, 0.394338, 0.105662, 0.394338],
                    [0.394338, 0.105662, 0.394338, 0.105662],
                ],
            ]
        )
        self._idx_face = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
