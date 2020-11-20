import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Tri3(ShapeFunction):
    """
    Tri3 (3-node triangular element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self.shape = 'TRI'
        self.name = 'TRI3'
        self.n_dof = 2
        self.n_node = 3
        self.n_intgp = 1
        self.n_face = 3
        self.n_fnode = 2
        self.weight = np.array([0.5])
        self.Shpfnc = np.array([
            [0.333333333333333],
            [0.333333333333333],
            [0.333333333333333]
        ])
        self.Bmatrix_nat = np.array([
            [
                [1.],
                [0.],
                [-1.]
            ],
            [
                [0.],
                [1.],
                [-1.]
            ]
        ])
        self.idx_face = np.array([
            [1, 2],
            [2, 0],
            [0, 1]
        ])
