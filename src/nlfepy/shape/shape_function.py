import numpy as np
from typing import Tuple


class ShapeFunction():
    """
    Base class of shape function

    Attributes
    ----------
    shape : str
        Shape of finite element (Tri, Quad, ...)
    name : str
        Name of finite element (Tri3, Tri6, Quad4, ...)
    n_dof : int
        Number of degrees of freedom
    n_node : int
        Number of nodes of each element
    n_intgp : int
        Number of integral points
    n_face : int
        Number of faces (3D) or edges (2D)
    n_fnode : int
        Number of nodes of each face (3D) or edge (2D)
    weight : float
        Weight of Gaussian integral
    Shpfnc : ndarray
        Shape function
    Bmatrix_nat : ndarray
        First derivative of shape function in natural coordinates
    idx_face : array-like
        Connectivity between face and node number: inod = idx_face[iface][ifnod]
    """

    def __init__(self) -> None:
        self.shape = None
        self.name = None
        self.n_dof = None
        self.n_node = None
        self.n_intgp = None
        self.n_face = None
        self.n_fnode = None
        self.weight = None
        self.Shpfnc = None
        self.Bmatrix_nat = None
        self.idx_face = None

    def get_bmatrix(self, cod: np.ndarray, iint: int) -> Tuple[np.ndarray, float]:
        """
        Get 1st derivative of shape function in global coordinates

        Parameters
        ----------
        cod : ndarray
            Coordinates of node in each element
        iint : int
            Index of integral point

        Returns
        -------
        shapef : tuple
            Bmatrix and weigh * determinant of Jacobi matrix
        """

        JacobT = np.dot(self.Bmatrix_nat[:, :, iint], cod.T)
        detJ = np.sqrt(np.linalg.det(np.dot(JacobT, JacobT.T)))
        Jinv = np.linalg.inv(JacobT)
        Bmatrix_phys = np.dot(Jinv, self.Bmatrix_nat[:, :, iint])
        wdetJ = self.weight[iint]*detJ
        return Bmatrix_phys, wdetJ

    def get_wdetJ(self, cod: np.ndarray, iint: int) -> float:
        """
        Get w*detJ

        Parameters
        ----------
        cod : ndarray
            Coordinates of node in each element
        iint : int
            Index of integral point

        Returns
        -------
        shapef : float
            Weigh * determinant of Jacobi matrix
        """

        Jacob = np.dot(cod, self.Bmatrix_nat[:, :, iint].T)
        detJ = np.sqrt(np.linalg.det(np.dot(Jacob.T, Jacob)))
        return self.weight[iint]*detJ
