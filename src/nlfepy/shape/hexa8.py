import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Hexa8(ShapeFunction):
    """
    Hexa8 (8-node hexahedral element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "HEXA"
        self._name = "HEXA8"
        self._n_dof = 3
        self._n_node = 8
        self._n_intgp = 8
        self._n_face = 6
        self._n_fnode = 4
        self._weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self._Shpfnc = np.array(
            [
                [
                    0.490563,
                    0.131446,
                    0.131446,
                    0.0352208,
                    0.131446,
                    0.0352208,
                    0.0352208,
                    0.00943739,
                ],
                [
                    0.131446,
                    0.490563,
                    0.0352208,
                    0.131446,
                    0.0352208,
                    0.131446,
                    0.00943739,
                    0.0352208,
                ],
                [
                    0.0352208,
                    0.131446,
                    0.131446,
                    0.490563,
                    0.00943739,
                    0.0352208,
                    0.0352208,
                    0.131446,
                ],
                [
                    0.131446,
                    0.0352208,
                    0.490563,
                    0.131446,
                    0.0352208,
                    0.00943739,
                    0.131446,
                    0.0352208,
                ],
                [
                    0.131446,
                    0.0352208,
                    0.0352208,
                    0.00943739,
                    0.490563,
                    0.131446,
                    0.131446,
                    0.0352208,
                ],
                [
                    0.0352208,
                    0.131446,
                    0.00943739,
                    0.0352208,
                    0.131446,
                    0.490563,
                    0.0352208,
                    0.131446,
                ],
                [
                    0.00943739,
                    0.0352208,
                    0.0352208,
                    0.131446,
                    0.0352208,
                    0.131446,
                    0.131446,
                    0.490563,
                ],
                [
                    0.0352208,
                    0.00943739,
                    0.131446,
                    0.0352208,
                    0.131446,
                    0.0352208,
                    0.490563,
                    0.131446,
                ],
            ]
        )
        self._Bmatrix_nat = np.array(
            [
                [
                    [
                        -0.311004,
                        -0.311004,
                        -0.0833333,
                        -0.0833333,
                        -0.0833333,
                        -0.0833333,
                        -0.0223291,
                        -0.0223291,
                    ],
                    [
                        0.311004,
                        0.311004,
                        0.0833333,
                        0.0833333,
                        0.0833333,
                        0.0833333,
                        0.0223291,
                        0.0223291,
                    ],
                    [
                        0.0833333,
                        0.0833333,
                        0.311004,
                        0.311004,
                        0.0223291,
                        0.0223291,
                        0.0833333,
                        0.0833333,
                    ],
                    [
                        -0.0833333,
                        -0.0833333,
                        -0.311004,
                        -0.311004,
                        -0.0223291,
                        -0.0223291,
                        -0.0833333,
                        -0.0833333,
                    ],
                    [
                        -0.0833333,
                        -0.0833333,
                        -0.0223291,
                        -0.0223291,
                        -0.311004,
                        -0.311004,
                        -0.0833333,
                        -0.0833333,
                    ],
                    [
                        0.0833333,
                        0.0833333,
                        0.0223291,
                        0.0223291,
                        0.311004,
                        0.311004,
                        0.0833333,
                        0.0833333,
                    ],
                    [
                        0.0223291,
                        0.0223291,
                        0.0833333,
                        0.0833333,
                        0.0833333,
                        0.0833333,
                        0.311004,
                        0.311004,
                    ],
                    [
                        -0.0223291,
                        -0.0223291,
                        -0.0833333,
                        -0.0833333,
                        -0.0833333,
                        -0.0833333,
                        -0.311004,
                        -0.311004,
                    ],
                ],
                [
                    [
                        -0.311004,
                        -0.0833333,
                        -0.311004,
                        -0.0833333,
                        -0.0833333,
                        -0.0223291,
                        -0.0833333,
                        -0.0223291,
                    ],
                    [
                        -0.0833333,
                        -0.311004,
                        -0.0833333,
                        -0.311004,
                        -0.0223291,
                        -0.0833333,
                        -0.0223291,
                        -0.0833333,
                    ],
                    [
                        0.0833333,
                        0.311004,
                        0.0833333,
                        0.311004,
                        0.0223291,
                        0.0833333,
                        0.0223291,
                        0.0833333,
                    ],
                    [
                        0.311004,
                        0.0833333,
                        0.311004,
                        0.0833333,
                        0.0833333,
                        0.0223291,
                        0.0833333,
                        0.0223291,
                    ],
                    [
                        -0.0833333,
                        -0.0223291,
                        -0.0833333,
                        -0.0223291,
                        -0.311004,
                        -0.0833333,
                        -0.311004,
                        -0.0833333,
                    ],
                    [
                        -0.0223291,
                        -0.0833333,
                        -0.0223291,
                        -0.0833333,
                        -0.0833333,
                        -0.311004,
                        -0.0833333,
                        -0.311004,
                    ],
                    [
                        0.0223291,
                        0.0833333,
                        0.0223291,
                        0.0833333,
                        0.0833333,
                        0.311004,
                        0.0833333,
                        0.311004,
                    ],
                    [
                        0.0833333,
                        0.0223291,
                        0.0833333,
                        0.0223291,
                        0.311004,
                        0.0833333,
                        0.311004,
                        0.0833333,
                    ],
                ],
                [
                    [
                        -0.311004,
                        -0.0833333,
                        -0.0833333,
                        -0.0223291,
                        -0.311004,
                        -0.0833333,
                        -0.0833333,
                        -0.0223291,
                    ],
                    [
                        -0.0833333,
                        -0.311004,
                        -0.0223291,
                        -0.0833333,
                        -0.0833333,
                        -0.311004,
                        -0.0223291,
                        -0.0833333,
                    ],
                    [
                        -0.0223291,
                        -0.0833333,
                        -0.0833333,
                        -0.311004,
                        -0.0223291,
                        -0.0833333,
                        -0.0833333,
                        -0.311004,
                    ],
                    [
                        -0.0833333,
                        -0.0223291,
                        -0.311004,
                        -0.0833333,
                        -0.0833333,
                        -0.0223291,
                        -0.311004,
                        -0.0833333,
                    ],
                    [
                        0.311004,
                        0.0833333,
                        0.0833333,
                        0.0223291,
                        0.311004,
                        0.0833333,
                        0.0833333,
                        0.0223291,
                    ],
                    [
                        0.0833333,
                        0.311004,
                        0.0223291,
                        0.0833333,
                        0.0833333,
                        0.311004,
                        0.0223291,
                        0.0833333,
                    ],
                    [
                        0.0223291,
                        0.0833333,
                        0.0833333,
                        0.311004,
                        0.0223291,
                        0.0833333,
                        0.0833333,
                        0.311004,
                    ],
                    [
                        0.0833333,
                        0.0223291,
                        0.311004,
                        0.0833333,
                        0.0833333,
                        0.0223291,
                        0.311004,
                        0.0833333,
                    ],
                ],
            ]
        )
        self._idx_face = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 1, 0],
                [1, 5, 6, 2],
                [6, 7, 3, 2],
                [4, 0, 3, 7],
                [5, 4, 7, 6],
            ]
        )
