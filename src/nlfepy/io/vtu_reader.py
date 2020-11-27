import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from logging import getLogger


class VtuReader:
    """
    VTU reader

    Read mesh info and set boundary conditions.
    """

    def __init__(self, mesh_path=None) -> None:

        self.logger = getLogger('LogReader')

        if mesh_path is not None:
            self.read(mesh_path)

    def read(self, mesh_path: str) -> None:
        """
        Read .vtu mesh file

        Parameters
        ----------
        mesh_path : str
            Mesh file path. Mesh file must be written in VTK XML format (.vtu).
        """

        if not os.path.isfile(mesh_path):
            self.logger.error('Cannot find mesh file: {}'.format(mesh_path))
            sys.exit(1)

        try:
            tree = ET.parse(mesh_path)
        except Exception:
            self.logger.error('Cannot parse mesh file: {}'.format(mesh_path))
            sys.exit(1)

        root = tree.getroot()
        unstructured_grid = root.find(r'{VTK}UnstructuredGrid')
        piece = unstructured_grid.find(r'{VTK}Piece')

        # Mesh info.
        self.mesh = {}
        self.mesh['n_element'] = int(piece.attrib[r'NumberOfCells'])
        self.mesh['n_point'] = int(piece.attrib[r'NumberOfPoints'])
        self.mesh['coords'] = self.read_coordinates(piece=piece)
        self.mesh['connectivity'], self.mesh['n_node'], self.mesh['m_node'] = self.read_connectivity(piece=piece)
        if np.max(np.abs(self.mesh['coords'][2])) > 1.e-30:
            self.mesh['n_dof'] = 3
        else:
            self.mesh['n_dof'] = 2

        # Point data
        point_data = piece.find(r'{VTK}PointData')
        pdata_array = point_data.findall(r'{VTK}DataArray')

        # Cell data
        cell_data = piece.find(r'{VTK}CellData')
        cell_array = cell_data.findall(r'{VTK}DataArray')

        # Boundary condition
        self.bc_table = self.get_value(pdata_array, r'Boundary Condition').astype(np.int)
        self.prescribed_displacement = self.get_value(pdata_array, r'Prescribed Displacement').astype(np.float)
        self.prescribed_traction = self.get_value(pdata_array, r'Prescribed Traction').astype(np.float)
        self.applied_force = self.get_value(pdata_array, r'Applied Force').astype(np.float)
        self.set_boundary_condition()

        # Multi point constraint
        self.mpc_table = self.get_value(pdata_array, r'Multi-Point Constraints').astype(np.int)
        self.mpc_ratio = self.get_value(pdata_array, r'MPC Ratio').astype(np.float)
        self.set_multi_point_constraint()

        # Sub-structure
        self.mesh['grain_numbers'] = self.get_value(cell_array, r'Cryst')[:, 0].astype(np.int)
        self.mesh['material_numbers'] = self.get_value(cell_array, r'Mater')[:, 0].astype(np.int)
        self.mesh['crystal_orientation'] = self.get_value(cell_array, r'Crystal Orientation').astype(np.float)

    def read_coordinates(self, *, piece):
        """
        Read coordinate of mesh file
        """

        points = piece.find(r'{VTK}Points')
        cods = points.find(r'{VTK}DataArray')

        coords_list = cods.text.split('\n')
        coords = []
        for cod_lst in coords_list:
            if cod_lst == '':
                continue
            cod = cod_lst.split(' ')
            c_lst = []
            for idof in range(3):
                c_lst.append(float(cod[idof]))
            coords.append(c_lst)

        coords = np.array(coords).T

        return coords

    def read_connectivity(self, *, piece):
        """
        Read and set connectivity between element and nodes
        """

        cells = piece.find(r'{VTK}Cells')
        darray = cells.findall(r'{VTK}DataArray')
        lnodes = []
        n_node = []
        m_node = 0
        for d in darray:
            if d.attrib[r'Name'] == 'connectivity':
                lnodes_list = d.text.split('\n')
                for lnd_lst in lnodes_list:
                    if lnd_lst == '':
                        continue
                    lnd = lnd_lst.split(' ')
                    nod_lst = []
                    for nod in lnd:
                        if nod != '':
                            nod_lst.append(int(nod))
                    lnodes.append(nod_lst)
                    n_node.append(len(nod_lst))
                    m_node = max(m_node, n_node[-1])

        return lnodes, n_node, m_node

    def get_value(self, darray, tag):

        vals = []
        for d in darray:
            if d.attrib[r'Name'] == tag:
                val_list = d.text.split('\n')
                for vlst in val_list:
                    if vlst == '':
                        continue
                    vl = vlst.split(' ')
                    vs = []
                    for v in vl:
                        if vars != '':
                            vs.append(v)
                    vals.append(vs)

        vals = np.array(vals)

        return vals

    def set_boundary_condition(self):
        """
        Set boundary conditions (Fix point, prescribed displacement, ...)
        """

        FIX, DISPLACEMENT, LOAD, FORCE = 1, -1, -2, -3

        self.bc = {}
        self.bc['n_fix'] = np.count_nonzero(self.bc_table == FIX) + np.count_nonzero(self.bc_table == DISPLACEMENT)

        ipnt, idof = np.where(self.bc_table == FIX)
        jpnt, jdof = np.where(self.bc_table == DISPLACEMENT)
        idx_i = self.mesh['n_dof']*ipnt + idof
        idx_j = self.mesh['n_dof']*jpnt + jdof
        self.bc['idx_fix'] = np.concatenate([idx_i, idx_j])

        n_displacement = np.count_nonzero(self.bc_table == DISPLACEMENT)
        n_load = np.count_nonzero(self.bc_table == LOAD)
        n_force = np.count_nonzero(self.bc_table == FORCE)

        if n_displacement == 0 and n_load == 0 and n_force == 0:
            self.bc['type'] = 'MPC'
        else:
            self.bc['type'] = 'BC'

        self.bc['n_disp'] = n_displacement
        if self.bc['n_disp'] > 0:
            idx_pnt, self.bc['direction'] = np.where(self.bc_table == DISPLACEMENT)
            self.bc['idx_disp'] = self.mesh['n_dof']*idx_pnt + self.bc['direction']
            self.bc['displacement'] = self.prescribed_displacement[self.bc_table == DISPLACEMENT]
        else:
            self.bc['direction'] = self.bc['idx_disp'] = self.bc['displacement'] = []

        self.bc['traction'] = self.prescribed_traction
        self.bc['applied_force'] = self.applied_force

    def set_multi_point_constraint(self):
        """
        Set multi-point constraint for homogenization or unit cell analysis
        """

        self.mpc = {}
        self.mpc['nmpcpt'] = np.max(self.mpc_table)
        self.mpc['slave'] = []
        self.mpc['master'] = []
        self.mpc['ratio'] = []
        for impc in range(1, self.mpc['nmpcpt'] + 1):
            pnt, dof = np.where(self.mpc_table == impc)
            for ipnt, idof in zip(pnt, dof):
                ratio = np.abs(self.mpc_ratio[ipnt][idof])
                if ratio > 0.:
                    self.mpc['slave'].append(self.mesh['n_dof'] * ipnt + idof)
                    self.mpc['ratio'].append(ratio)
                else:
                    self.mpc['master'].append(self.mesh['n_dof'] * ipnt + idof)
