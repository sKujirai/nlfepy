import numpy as np
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
    # n_node : array-like
    #     Number of nodes of each element [n_element]
    # m_node : int
    #     Max number of n_node
    coords : ndarray
        Coordinates [n_dof, n_point]
    connectivity : list
        Connectivity of elaments [n_element][n_node] (2D array)
    element_name : list
        Name of each finite elemenet [n_element]
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
        self.n_dof = None
        self.n_point = None
        self.n_element = None
        self.n_dfdof = None
        # self.n_node = None
        # self.m_node = None
        self.coords = None
        self.connectivity = None
        self.element_name = None
        self.material_numbers = None
        self.grain_numbers = None
        self.crystal_orientation = None
        self.shapef = None
        self.bc = None

    def read(self, mesh_path: str) -> None:
        """
        Read mesh file and set mesh info

        Parameters
        ----------
        mesh_path : str
            Mesh file path
        """

        reader = VtuReader(mesh_path)
        self.set_mesh_dict(reader.mesh)
        self.set_mesh_info()
        self.set_boundary_condition(reader.bc)

    def set_shape(self, coords: np.ndarray, connectivity) -> None:
        """
        Set mesh shape from coordinates and connectivity

        Parameters
        ----------
        coords : ndarray
            Coordiantes [n_dof, n_point] (2D array)
        connectivity : array-like
            Connectivity of elements [n_element][n_node] (2D array)
        """

        self.coords = coords
        self.n_dof, self.n_point = coords.shape
        self.n_dfdof = 3 if self.n_dof == 2 else 6
        self.connectivity = connectivity
        self.n_element = len(connectivity)
        self.material_numbers = np.zeros(self.n_element)
        self.material_numbers = np.zeros(self.n_element)
        self.crystal_orientation = np.zeros((3, self.n_element))

        self.set_mesh_info()

    def set_mesh_dict(self, data: dict) -> None:
        """
        Set mesh info

        Parameters
        ----------
        data : dict
            Mesh info (keys: 'n_dof', 'n_point', 'n_element', 'n_node', 'coords', 'connectivity')
        """

        self.n_dof = data['n_dof']
        self.n_dfdof = 3 if self.n_dof == 2 else 6
        self.n_point = data['n_point']
        self.n_element = data['n_element']
        self.coords = data['coords'][:self.n_dof]
        self.connectivity = data['connectivity']
        if 'material_numbers' in data:
            self.material_numbers = data['material_numbers']
        else:
            self.material_numbers = np.zeros(self.n_element)
        if 'grain_numbers' in data:
            self.grain_numbers = data['grain_numbers']
        else:
            self.material_numbers = np.zeros(self.n_element)
        if 'crystal_orientation' in data:
            self.crystal_orientation = data['crystal_orientation']
        else:
            self.crystal_orientation = np.zeros((3, self.n_element))

    def set_mesh_info(self) -> None:

        # self.n_node = []
        # self.m_node = 0
        self.shapef = {}
        self.element_name = []

        for lnod in self.connectivity:
            n_nod = len(lnod)
            # self.n_node.append(n_nod)
            # self.m_node = max(self.m_node, n_nod)
            self.element_name.append(get_element_name(n_dof=self.n_dof, n_node=n_nod))

        for elm_name in set(self.element_name):
            self.shapef[elm_name] = get_shape_function(elm_name)

    def set_boundary_condition(self, bc: dict) -> None:
        self.bc = bc

    def get_Shpfnc(self, etype: str, *, elm: int) -> np.ndarray:
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

        return self.shapef[self.element_name[elm]][etype].Shpfnc

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
            cod = self.coords[:, np.array(self.connectivity[elm])[nds]][:self.n_dof]
        else:
            cod = self.coords[:, self.connectivity[elm]][:self.n_dof]

        return self.shapef[self.element_name[elm]][etype].get_Nmatrix(cod, itg)

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
            cod = self.coords[:, np.array(self.connectivity[elm])[nds]][:self.n_dof]
        else:
            cod = self.coords[:, self.connectivity[elm]][:self.n_dof]

        return self.shapef[self.element_name[elm]][etype].get_Bmatrix(cod, itg)

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

        return self.shapef[self.element_name[elm]][etype].n_intgp

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

        return self.shapef[self.element_name[elm]][etype].n_node

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

        return self.shapef[self.element_name[elm]][etype].idx_face
