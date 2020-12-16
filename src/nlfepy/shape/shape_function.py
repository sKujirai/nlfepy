from abc import ABCMeta
import numpy as np
from typing import Tuple


class ShapeFunction(metaclass=ABCMeta):
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
        self._shape = None
        self._name = None
        self._n_dof = None
        self._n_node = None
        self._n_intgp = None
        self._n_face = None
        self._n_fnode = None
        self._weight = None
        self._Shpfnc = None
        self._Bmatrix_nat = None
        self._idx_face = None

    @property
    def shape(self) -> str:
        return self._shape

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def n_node(self) -> int:
        return self._n_node

    @property
    def n_intgp(self) -> int:
        return self._n_intgp

    @property
    def n_face(self) -> int:
        return self._n_face

    @property
    def n_fnode(self) -> int:
        return self._n_fnode

    @property
    def Shpfnc(self) -> np.ndarray:
        return self._Shpfnc

    @property
    def idx_face(self) -> np.ndarray:
        return self._idx_face

    def get_Bmatrix(self, cod: np.ndarray, iint: int) -> Tuple[np.ndarray, float]:
        """
        Get 1st derivative of shape function (B-matrix) in global coordinates

        Parameters
        ----------
        cod : ndarray
            Coordinates of node in each element
        iint : int
            Index of integral point

        Returns
        -------
        shapef : tuple
            B-matrix and weigh * determinant of Jacobi matrix
        """

        JacobT = np.dot(self._Bmatrix_nat[:, :, iint], cod.T)
        detJ = np.sqrt(np.linalg.det(np.dot(JacobT, JacobT.T)))
        Jinv = np.linalg.inv(JacobT)
        Bmatrix_phys = np.dot(Jinv, self._Bmatrix_nat[:, :, iint])
        wdetJ = self._weight[iint] * detJ
        return Bmatrix_phys, wdetJ

    def get_Nmatrix(self, cod: np.ndarray, iint: int) -> Tuple[np.ndarray, float]:
        """
        Get shape function (N-matrix) and w*detJ

        Parameters
        ----------
        cod : ndarray
            Coordinates of node in each element
        iint : int
            Index of integral point

        Returns
        -------
        shapef : tuple
            N-matrix and weigh * determinant of Jacobi matrix
        """

        return self._Shpfnc, self.get_wdetJ(cod, iint)

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

        Jacob = np.dot(cod, self._Bmatrix_nat[:, :, iint].T)
        detJ = np.sqrt(np.linalg.det(np.dot(Jacob.T, Jacob)))
        return self._weight[iint] * detJ
