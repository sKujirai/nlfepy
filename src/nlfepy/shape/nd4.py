import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Nd4(ShapeFunction):
    """
    Nd4 (4-node element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "ND"
        self._name = "ND4"
        self._n_dof = 1
        self._n_node = 4
        self._n_intgp = 3
        self._weight = np.array(
            [
                0.555555555555556,
                0.888888888888889,
                0.555555555555556,
            ]
        )
        self._Shpfnc = np.array(
            [
                [0.488014084041407, -0.062500000000000, 0.061985915958592],
                [0.061985915958592, -0.062500000000000, 0.488014084041407],
                [0.747852751738002, 0.562500000000000, -0.297852751738002],
                [-0.297852751738002, 0.562500000000000, 0.747852751738002],
            ]
        )
        self._Bmatrix_nat = np.array(
            [
                [
                    [-1.821421252896667, 0.062500000000000, -0.078578747103331],
                    [0.078578747103331, -0.062500000000000, 1.821421252896667],
                    [2.221421252896665, -1.687500000000000, 0.478578747103328],
                    [-0.478578747103328, 1.687500000000000, -2.221421252896665],
                ]
            ]
        )
