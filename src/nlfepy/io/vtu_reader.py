import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from logging import getLogger
from typing import Tuple, Optional, List, Union


class VtuReader:
    """
    VTU reader

    Read mesh info and set boundary conditions.
    """

    def __init__(self, mesh_path: str = None) -> None:

        self._mesh: dict = {}
        self._bc: Optional[dict] = None
        self._mpc: Optional[dict] = None
        self._pdata_array: Optional[List[ET.Element]] = None
        self._cdata_array: Optional[List[ET.Element]] = None

        self._logger = getLogger("LogReader")

        if mesh_path is not None:
            self.read(mesh_path)

    @property
    def mesh(self) -> dict:
        if self._mesh is None:
            self._logger.error("Mesh object is not set")
            sys.exit(1)
        return self._mesh

    @property
    def bc(self) -> Optional[dict]:
        return self._bc

    @property
    def mpc(self) -> Optional[dict]:
        return self._mpc

    def read(self, mesh_path: str) -> None:
        """
        Read .vtu mesh file

        Parameters
        ----------
        mesh_path : str
            Mesh file path. Mesh file must be written in VTK XML format (.vtu).
        """

        if not os.path.isfile(mesh_path):
            self._logger.error("Cannot find mesh file: {}".format(mesh_path))
            sys.exit(1)

        try:
            tree = ET.parse(mesh_path)
        except Exception:
            self._logger.error("Cannot parse mesh file: {}".format(mesh_path))
            sys.exit(1)

        root = tree.getroot()
        unstructured_grid = root.find(r"{VTK}UnstructuredGrid")
        if unstructured_grid is None:
            self._logger.error("Cannot get element of unstructured grid")
            sys.exit(1)
        piece = unstructured_grid.find(r"{VTK}Piece")
        if piece is None:
            self._logger.error("Cannot get value in unstructured grid")
            sys.exit(1)

        # Mesh info.
        self._mesh = {}
        self._mesh["n_element"] = int(piece.attrib[r"NumberOfCells"])
        self._mesh["n_point"] = int(piece.attrib[r"NumberOfPoints"])
        self._mesh["coords"] = self._read_coordinates(piece=piece)
        (
            self._mesh["connectivity"],
            self._mesh["n_node"],
            self._mesh["m_node"],
        ) = self._read_connectivity(piece=piece)
        if np.max(np.abs(self._mesh["coords"][2])) > 1.0e-30:
            self._mesh["n_dof"] = 3
        else:
            self._mesh["n_dof"] = 2

        # Point data
        point_data = piece.find(r"{VTK}PointData")
        if point_data is not None:
            self._pdata_array = point_data.findall(r"{VTK}DataArray")
            self._keys_pnt = [dat.attrib["Name"] for dat in self._pdata_array]

            # Boundary condition
            self._bc_table = self._get_value(
                self._pdata_array, r"Boundary Condition"
            ).astype(np.int)
            self._prescribed_displacement = self._get_value(
                self._pdata_array, r"Prescribed Displacement"
            ).astype(np.float)
            self._prescribed_traction = self._get_value(
                self._pdata_array, r"Prescribed Traction"
            ).astype(np.float)
            self._applied_force = self._get_value(
                self._pdata_array, r"Applied Force"
            ).astype(np.float)
            if self._bc_table.shape[0] > 0:
                self._set_boundary_condition()

            # Multi point constraint
            self._mpc_table = self._get_value(
                self._pdata_array, r"Multi-Point Constraints"
            ).astype(np.int)
            self._mpc_ratio = self._get_value(self._pdata_array, r"MPC Ratio").astype(
                np.float
            )
            if self._mpc_table.shape[0] > 0:
                self._set_multi_point_constraint()

        # Cell data
        cell_data = piece.find(r"{VTK}CellData")
        if cell_data is not None:
            self._mesh["mesh_type"] = "FiniteElement"

            self._cell_array = cell_data.findall(r"{VTK}DataArray")
            self._keys_cell = [dat.attrib["Name"] for dat in self._cell_array]
 
            # Sub-structure
            self._mesh["grain_numbers"] = self._get_value(self._cell_array, r"Cryst")[
                :, 0
            ].astype(np.int)
            self._mesh["material_numbers"] = self._get_value(self._cell_array, r"Mater")[
                :, 0
            ].astype(np.int)
            self._mesh["crystal_orientation"] = self._get_value(
                self._cell_array, r"Crystal Orientation"
            ).astype(np.float)
        else:
            self._mesh["mesh_type"] = "Meshfree"

            if point_data is not None:
                # Sub-structure
                self._mesh["grain_numbers"] = self._get_value(self._pdata_array, r"Cryst")[
                    :, 0
                ].astype(np.int)
                self._mesh["material_numbers"] = self._get_value(self._pdata_array, r"Mater")[
                    :, 0
                ].astype(np.int)
                self._mesh["crystal_orientation"] = self._get_value(
                    self._pdata_array, r"Crystal Orientation"
                ).astype(np.float)

    def get_elm_value(
        self, tag: str, *, systems: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get element value
        """

        if self._cell_array is None:
            self._logger.error("No cell data is found")
            sys.exit(1)

        if tag not in self._keys_cell:
            self._logger.error("Cannot find value {} in cell data".format(tag))
            sys.exit(1)

        value = self._get_value(self._cell_array, tag)

        if systems is None:
            return value
        else:
            if isinstance(systems, list):
                systems = [s - 1 for s in systems]
            else:
                systems -= 1
                if type(systems) is int:
                    systems = [systems]
            return value[:, systems]

    def get_point_value(
        self, tag: str, *, systems: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Get nodal value
        """

        if self._pdata_array is None:
            self._logger.error("No point data is found")
            sys.exit(1)

        if tag not in self._keys_pnt:
            self._logger.error("Cannot find value {} in point data".format(tag))
            sys.exit(1)

        value = self._get_value(self._pdata_array, tag)

        if systems is None:
            return value
        else:
            if isinstance(systems, list):
                systems = [s - 1 for s in systems]
            else:
                systems -= 1
                if type(systems) is int:
                    systems = [systems]
            return value[:, systems]

    def _read_coordinates(self, *, piece) -> np.ndarray:
        """
        Read coordinate of mesh file
        """

        points = piece.find(r"{VTK}Points")
        cods = points.find(r"{VTK}DataArray")

        coords_list = cods.text.split("\n")
        coords = []
        for cod_lst in coords_list:
            if cod_lst == "":
                continue
            cod = cod_lst.split(" ")
            c_lst = []
            for idof in range(3):
                c_lst.append(float(cod[idof]))
            coords.append(c_lst)

        coords = np.array(coords).T

        return coords

    def _read_connectivity(self, *, piece) -> Tuple[List[List[int]], List[int], int]:
        """
        Read and set connectivity between element and nodes
        """

        cells = piece.find(r"{VTK}Cells")
        darray = cells.findall(r"{VTK}DataArray")
        lnodes = []
        n_node = []
        m_node = 0
        for d in darray:
            if d.attrib[r"Name"] == "connectivity":
                lnodes_list = d.text.split("\n")
                for lnd_lst in lnodes_list:
                    if lnd_lst == "":
                        continue
                    lnd = lnd_lst.split(" ")
                    nod_lst = []
                    for nod in lnd:
                        if nod != "":
                            nod_lst.append(int(nod))
                    lnodes.append(nod_lst)
                    n_node.append(len(nod_lst))
                    m_node = max(m_node, n_node[-1])

        return lnodes, n_node, m_node

    def _get_value(self, darray, tag) -> np.ndarray:

        vals = []
        for d in darray:
            if d.attrib[r"Name"] == tag:
                val_list = d.text.split("\n")
                for vlst in val_list:
                    if vlst == "":
                        continue
                    vl = vlst.split(" ")
                    vs = []
                    for v in vl:
                        if v != "":
                            vs.append(v)
                    vals.append(vs)

        vals = np.array(vals)

        return vals

    def _set_boundary_condition(self) -> None:
        """
        Set boundary conditions (Fix point, prescribed displacement, ...)
        """

        FIX, DISPLACEMENT, LOAD, FORCE = 1, -1, -2, -3

        self._bc = {}
        self._bc["n_fix"] = np.count_nonzero(self._bc_table == FIX) + np.count_nonzero(
            self._bc_table == DISPLACEMENT
        )

        ipnt, idof = np.where(self._bc_table == FIX)
        jpnt, jdof = np.where(self._bc_table == DISPLACEMENT)
        idx_i = self._mesh["n_dof"] * ipnt + idof
        idx_j = self._mesh["n_dof"] * jpnt + jdof
        self._bc["idx_fix"] = np.concatenate([idx_i, idx_j])

        n_displacement = np.count_nonzero(self._bc_table == DISPLACEMENT)
        n_load = np.count_nonzero(self._bc_table == LOAD)
        n_force = np.count_nonzero(self._bc_table == FORCE)

        if n_displacement == 0 and n_load == 0 and n_force == 0:
            self._bc["type"] = "MPC"
        else:
            self._bc["type"] = "BC"

        self._bc["n_disp"] = n_displacement
        if self._bc["n_disp"] > 0:
            idx_pnt, self._bc["direction"] = np.where(self._bc_table == DISPLACEMENT)
            self._bc["idx_disp"] = self._mesh["n_dof"] * idx_pnt + self._bc["direction"]
            self._bc["displacement"] = self._prescribed_displacement[
                self._bc_table == DISPLACEMENT
            ]
        else:
            self._bc["direction"] = self._bc["idx_disp"] = self._bc["displacement"] = []

        self._bc["traction"] = self._prescribed_traction
        self._bc["applied_force"] = self._applied_force

    def _set_multi_point_constraint(self) -> None:
        """
        Set multi-point constraint for homogenization or unit cell analysis
        """

        self._mpc = {}
        self._mpc["nmpcpt"] = np.max(self._mpc_table)
        self._mpc["slave"] = []
        self._mpc["master"] = []
        self._mpc["ratio"] = []
        for impc in range(1, self._mpc["nmpcpt"] + 1):
            pnt, dof = np.where(self._mpc_table == impc)
            for ipnt, idof in zip(pnt, dof):
                ratio = np.abs(self._mpc_ratio[ipnt][idof])
                if ratio > 0.0:
                    self._mpc["slave"].append(self._mesh["n_dof"] * ipnt + idof)
                    self._mpc["ratio"].append(ratio)
                else:
                    self._mpc["master"].append(self._mesh["n_dof"] * ipnt + idof)
