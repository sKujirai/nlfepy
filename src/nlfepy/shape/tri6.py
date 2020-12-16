import numpy as np
from typing import Tuple
from .shape_function import ShapeFunction


class Tri6(ShapeFunction):
    """
    Tri3 (3-node triangular element) class inheriting class: ShapeFunction
    """

    def __init__(self) -> None:

        super().__init__()

        self._shape = "TRI"
        self._name = "TRI6"
        self._n_dof = 2
        self._n_node = 6
        self._n_intgp = 3
        self._n_face = 3
        self._n_fnode = 3
        self._weight = np.array(
            [0.166666666666667, 0.166666666666667, 0.166666666666667]
        )
        self._Shpfnc = np.array(
            [
                [-0.11111111111111123, 0.2222222222222227, -0.11111111111111123],
                [-0.11111111111111123, -0.11111111111111123, 0.2222222222222227],
                [0.22222222222222124, -0.1111111111111109, -0.11111111111111091],
                [0.11111111111111155, 0.44444444444444553, 0.44444444444444553],
                [0.4444444444444449, 0.11111111111111091, 0.4444444444444431],
                [0.4444444444444449, 0.444444444444443, 0.11111111111111094],
            ]
        )
        self._Bmatrix_nat = np.array(
            [
                [
                    [-0.33333333333333204, 1.6666666666666679, -0.33333333333333204],
                    [0.0, 0.0, 0.0],
                    [-1.6666666666666643, 0.3333333333333358, 0.3333333333333357],
                    [0.666666666666668, 0.666666666666668, 2.666666666666668],
                    [-0.666666666666668, -0.666666666666668, -2.666666666666668],
                    [1.9999999999999964, -2.0000000000000036, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [-0.33333333333333204, -0.33333333333333204, 1.6666666666666679],
                    [-1.6666666666666643, 0.3333333333333358, 0.3333333333333357],
                    [0.666666666666668, 2.666666666666668, 0.666666666666668],
                    [1.9999999999999964, 0.0, -2.0000000000000036],
                    [-0.666666666666668, -2.666666666666668, -0.666666666666668],
                ],
            ]
        )
        self._idx_face = np.array([[1, 2, 4], [2, 0, 5], [0, 1, 3]])
