import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Nd2(ShapeFunction):
    """
    Nd2 (2-node element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self.shape = 'ND'
        self.name = 'ND2'
        self.n_dof = 1
        self.n_node = 2
        self.n_intgp = 2
        self.weight = np.array([1., 1.])
        self.Shpfnc = np.array([
            [0.788675134594813, 0.211324865405187],
            [0.211324865405187, 0.788675134594813],
        ])
        self.Bmatrix_nat = np.array([
            [
                [-0.5, -0.5],
                [0.5, 0.5]
            ]
        ])
