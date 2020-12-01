from os import read
import sys
import numpy as np
from logging import getLogger
from typing import Tuple
from ..io.vtu_reader import VtuReader
from ..shape.shapef_util import get_shape_function, get_element_name


class Mesh:
    """
    Mesh class

    Attributes
    ----------
    n_dof : int
        Number of degrees of freedom
    n_point : int
        Number of nodes
    n_element : int
        Number of finite elements
    n_tintgp : int
        Number of total integral points
    itg_idx : array-like
        Index of total integral points
    coords : ndarray
        Coordinates [n_dof, n_point]
    connectivity : list
        Connectivity of elaments [n_element][n_node] (2D array)
    element_name : list
        Name of each finite elemenet [n_element]
    element_shape : list
        Shape of each finite elemenet [n_element]
    material_numbers : array-like
        Material numbers [n_element]
    grain_numbers : array-like
        Grain numbers [n_element]
    crystal_orientation : ndarray
        Crystal orientation [3, n_element]
    shapef : list
        List of shape functions
    bc : dict
        Boundary conditions
    """

    def __init__(self) -> None:
        self._n_dof = None
        self._n_point = None
        self._n_element = None
        self._n_dfdof = None
        self._n_tintgp = None
        self._itg_idx = None
        self._coords = None
        self._connectivity = None
        self._element_name = None
        self._element_shape = None
        self._material_numbers = None
        self._grain_numbers = None
        self._crystal_orientation = None
        self._shapef = None
        self._bc = None
        self._mpc = None

    @property
    def n_dof(self) -> int:
        return self._n_dof

    @property
    def n_point(self) -> int:
        return self._n_point

    @property
    def n_element(self) -> int:
        return self._n_element

    @property
    def n_dfdof(self) -> int:
        return self._n_dfdof

    @property
    def n_tintgp(self) -> int:
        return self._n_tintgp

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def connectivity(self) -> list:
        return self._connectivity

    @property
    def element_name(self) -> list:
        return self._element_name

    @property
    def element_shape(self) -> list:
        return self._element_shape

    @property
    def material_numbers(self):
        return self._material_numbers

    @property
    def grain_numbers(self):
        return self._grain_numbers

    @property
    def crystal_orientation(self) -> np.ndarray:
        return self._crystal_orientation

    @property
    def bc(self) -> dict:
        return self._bc

    @property
    def mpc(self) -> dict:
        return self._mpc

    def itg_idx(self, *, elm, itg) -> int:
        return self._itg_idx[elm] + itg

    def read(self, mesh_path: str) -> None:
        """
        Read mesh file and set mesh info

        Parameters
        ----------
        mesh_path : str
            Mesh file path
        """

        reader = VtuReader(mesh_path)
        self._set_mesh_dict(reader.mesh)
        self._set_mesh_info()
        self._set_bc_dict(bc=reader.bc, mpc=reader.mpc)

    def set_shape(self, *, coords: np.ndarray, connectivity) -> None:
        """
        Set mesh shape from coordinates and connectivity

        Parameters
        ----------
        coords : ndarray
            Coordiantes [n_dof, n_point] (2D array)
        connectivity : array-like
            Connectivity of elements [n_element][n_node] (2D array)
        """

        self._coords = coords
        self._n_dof, self._n_point = coords.shape
        if self._n_dof == 3:
            if np.max(np.abs(coords[2])) < 1.e-30:
                self._n_dof = 2
        self._n_dfdof = 3 if self._n_dof == 2 else 6
        if type(connectivity) is np.ndarray:
            connectivity = connectivity.tolist()
        self._connectivity = connectivity
        self._n_element = len(connectivity)
        self._material_numbers = np.zeros(self._n_element, dtype='int')
        self._grain_numbers = np.zeros(self._n_element, dtype='int')
        self._crystal_orientation = np.zeros((self._n_element, 3))

        self._set_mesh_info()

    def _set_mesh_dict(self, data: dict) -> None:
        """
        Set mesh info

        Parameters
        ----------
        data : dict
            Mesh info (keys: 'n_dof', 'n_point', 'n_element', 'n_node', 'coords', 'connectivity')
        """

        self._n_dof = data['n_dof']
        self._n_dfdof = 3 if self._n_dof == 2 else 6
        self._n_point = data['n_point']
        self._n_element = data['n_element']
        self._coords = data['coords'][:self._n_dof]
        self._connectivity = data['connectivity']
        if 'material_numbers' in data:
            self._material_numbers = data['material_numbers']
        else:
            self._material_numbers = np.zeros(self._n_element)
        if 'grain_numbers' in data:
            self._grain_numbers = data['grain_numbers']
        else:
            self._material_numbers = np.zeros(self._n_element)
        if 'crystal_orientation' in data:
            self._crystal_orientation = data['crystal_orientation']
        else:
            self._crystal_orientation = np.zeros((3, self._n_element))

    def _set_mesh_info(self) -> None:

        self._shapef = {}
        self._element_name = []

        for lnod in self._connectivity:
            n_nod = len(lnod)
            self._element_name.append(get_element_name(n_dof=self._n_dof, n_node=n_nod))

        for elm_name in set(self._element_name):
            self._shapef[elm_name] = get_shape_function(elm_name)

        self._n_tintgp = 0
        self._element_shape = []
        self._itg_idx = []
        for ielm in range(self._n_element):
            self._itg_idx.append(self._n_tintgp)
            self._n_tintgp += self._shapef[self._element_name[ielm]]['vol'].n_intgp
            self._element_shape.append(self._shapef[self._element_name[ielm]]['vol'].shape)

    def _set_bc_dict(self, *, bc: dict, mpc: dict) -> None:
        self._bc = bc
        self._mpc = mpc

    def set_bc(self, *, constraint='compression', model='full', value=None) -> None:
        """
        Set boundary conditions

        Parameters
        ----------
        constraint : str
            B.C. type ('compression', 'tensile', 'shear', 'load')
        model : str
            Type of analysis model ('full', 'quarter')
        value : ndarray
            Magnitude of prescribed displacement or traction
        """

        logger = getLogger('bc')

        self._bc = {}
        self._bc['type'] = 'BC'
        # if constraint in ['compression', 'tensile', 'shear']:
        #     self._bc['type'] = 'displacement'
        # elif constraint in ['load']:
        #     self._bc['type'] = 'load'
        # else:
        #     logger.error('Invalid constraint type: {}'.format(constraint))
        #     sys.exit(1)

        self._bc['displacement'] = []
        self._bc['traction'] = np.zeros((self._n_point, 3))

        cod_min = np.min(self._coords, axis=1)
        cod_max = np.max(self._coords, axis=1)
        specimen_size = cod_max - cod_min
        eps = min(specimen_size)*1.e-10

        # bottom
        idx_bottom = np.where(self._coords[1] < cod_min[1] + eps)[0]
        idx_top = np.where(self._coords[1] > cod_max[1] - eps)[0]
        idx_left = np.where(self._coords[0] < cod_min[0] + eps)[0]
        idx_fix = np.intersect1d(idx_bottom, idx_left)
        if self._n_dof == 3:
            idx_front = np.where(self._coords[2] < cod_min[2] + eps)[0]
            idx_fix = np.intersect1d(idx_fix, idx_front)

        idx_tdof_bottom = self._n_dof * idx_bottom + 1
        idx_tdof_top = self._n_dof * idx_top + 1
        idx_tdof_left = self._n_dof * idx_left
        idx_tdof_fix = self._n_dof * idx_fix

        if constraint in ['compression', 'tensile', 'load']:
            self._bc['idx_disp'] = np.sort(idx_tdof_top)
            self._bc['idx_fix'] = np.concatenate([idx_tdof_bottom, idx_tdof_top])
            if model == 'quarter':
                self._bc['idx_fix'] = np.sort(np.concatenate([self._bc['idx_fix'], idx_tdof_left]))
            else:
                self._bc['idx_fix'] = np.sort(np.concatenate([self._bc['idx_fix'], idx_tdof_fix]))
            if self._n_dof == 3:
                idx_tdof_fix_3d = self._n_dof * idx_fix + 2
                self._bc['idx_fix'] = np.sort(np.concatenate([self._bc['idx_fix'], idx_tdof_fix_3d]))
            self._bc['n_fix'] = self._bc['idx_fix'].shape[0]
            self._bc['n_disp'] = self._bc['idx_disp'].shape[0]
            if constraint in ['compression', 'tensile']:
                signv = 1. if constraint == 'tensile' else -1.
                self._bc['displacement'] = np.full(self._bc['n_disp'], signv * abs(value))
            elif constraint in ['load']:
                self._bc['traction'][idx_top, 3] = value
            else:
                logger.error('Invalid constraint type: {}'.format(constraint))
                sys.exit(1)
        else:
            logger.error('Invalid constraint type: {}'.format(constraint))
            sys.exit(1)

        # MPC
        self._mpc = {}
        self._mpc['nmpcpt'] = 0
        self._mpc['slave'] = self._mpc['master'] = self._mpc['ratio'] = []

    def get_shpfnc(self, etype: str, *, elm: int) -> np.ndarray:
        """
        Get shape function (N-matrix)

        Parameters
        ----------
        etype : str
            Element type ('vol' or 'area')
        elm : int
            Index of element

        Returns
        -------
        shapef : ndarray
            N-matrix
        """

        return self._shapef[self._element_name[elm]][etype].Shpfnc

    def get_Nmatrix(self, etype: str, *, elm: int, itg: int, nds: list = None) -> Tuple[np.ndarray, float]:
        """
        Get shape function (N-matrix) and w*detJ

        Parameters
        ----------
        etype : str
            Element type ('vol' or 'area')
        elm : int
            Index of element
        ing : int
            Index of integral point
        nds : list
            List of nodes index ('area')

        Returns
        -------
        shapef : tuple
            N-matrix and weigh * determinant of Jacobi matrix
        """
        if etype == 'area':
            cod = self._coords[:, np.array(self._connectivity[elm])[nds]][:self._n_dof]
        else:
            cod = self._coords[:, self._connectivity[elm]][:self._n_dof]

        return self._shapef[self._element_name[elm]][etype].get_Nmatrix(cod, itg)

    def get_Bmatrix(self, etype: str, *, elm: int, itg: int, nds: list = None) -> Tuple[np.ndarray, float]:
        """
        Get 1st derivative of shape function (B-matrix) in global coordinates

        Parameters
        ----------
        etype : str
            Element type ('vol' or 'area')
        elm : int
            Index of element
        ing : int
            Index of integral point
        nds : list
            List of nodes index ('area')

        Returns
        -------
        shapef : tuple
            B-matrix and weigh * determinant of Jacobi matrix
        """
        if etype == 'area':
            cod = self._coords[:, np.array(self._connectivity[elm])[nds]][:self._n_dof]
        else:
            cod = self._coords[:, self._connectivity[elm]][:self._n_dof]

        return self._shapef[self._element_name[elm]][etype].get_Bmatrix(cod, itg)

    def n_intgp(self, etype: str, *, elm) -> int:
        """
        Get number of integral points

        Parameters
        ----------
        etype : str
            Element type ('vol' or 'area')
        elm : int
            Index of element

        Returns
        -------
        n_intgp : int
            Number of integral point
        """

        return self._shapef[self._element_name[elm]][etype].n_intgp

    def n_node(self, etype: str, *, elm) -> int:
        """
        Get number of integral points

        Parameters
        ----------
        etype : str
            Element type ('vol' or 'area')
        elm : int
            Index of element

        Returns
        -------
        n_node : int
            Number of nodes
        """

        return self._shapef[self._element_name[elm]][etype].n_node

    def idx_face(self, etype: str, *, elm) -> list:
        """
        Get connectivity between face and nodes

        Parameters
        ----------
        etype : str
            Element type ('vol' or 'area')
        elm : int
            Index of element

        Returns
        -------
        idx_face : list
            Connectivity between face and nodes [face][node]
        """

        return self._shapef[self._element_name[elm]][etype].idx_face
