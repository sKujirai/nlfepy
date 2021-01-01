import sys
import numpy as np
from logging import getLogger


class VtuWriter:
    """
    VTU Writer

    Write mesh info and physical quantities
    """

    def __init__(self, *, mesh, values: dict = {}) -> None:

        self._mesh = mesh
        self._values = values

        self._EPS_MIN = 1.0e-30

        self._logger = getLogger("LogWriter")

    def _rearrange_node_numbers(self, nods: list, elm_name: str) -> list:
        """
        Rearrange node numbers for VTK file

        Parameters
        ----------
        nods : list
            Node numbers that consist finite element
        elm_name : str
            Finite element name

        Returns
        -------
        nods_rearranged : list
            Rearranged node numbers
        """

        if elm_name == "TRI6":
            return np.array(nods)[[0, 3, 1, 4, 2, 5]].tolist()
        elif elm_name == "QUAD8":
            return np.array(nods)[[0, 4, 1, 5, 2, 6, 3, 7]].tolist()
        else:
            return nods

    def _get_cell_type(self, elm_name: str) -> str:
        """
        Get VTK cell type

        Parameters
        ----------
        elm_name : str
            Finite element name

        Returns
        -------
        cell_type : str
            VTK cell type
        """

        if elm_name in ["TRI3"]:
            return "5"
        elif elm_name in ["QUAD4"]:
            return "9"
        elif elm_name in ["TRI6", "QUAD8"]:
            return "7"
        elif elm_name in ["TET4"]:
            return "10"
        elif elm_name in ["HEXA8"]:
            return "12"
        else:
            self._logger.error("Invalid finite element: {}".format(elm_name))
            sys.exit(1)

    def _write_bc(self, f) -> None:

        # B.C.
        FIX, DISPLACEMENT, LOAD, FORCE = 1, -1, -2, -3
        bc_table = np.zeros((3, self._mesh.n_point), dtype="int8")
        idx_cmp_fix = self._mesh.bc["idx_fix"]
        # Displacement
        disp_pts = None
        disp_dof = None
        if self._mesh.bc["n_disp"] > 0:
            disp_pts, disp_dof = np.divmod(self._mesh.bc["idx_disp"], self._mesh.n_dof)
            bc_table[disp_dof, disp_pts] = DISPLACEMENT
            idx_cmp_fix = np.setdiff1d(
                self._mesh.bc["idx_fix"], self._mesh.bc["idx_disp"]
            )
        # Fix point
        fix_pts, fix_dof = np.divmod(idx_cmp_fix, self._mesh.n_dof)
        bc_table[fix_dof, fix_pts] = FIX
        # Traction
        if "traction" in self._mesh.bc:
            traction = np.float32(self._mesh.bc["traction"])
            if traction.shape[1] == 2:
                traction = np.insert(traction, 2, 0.0, axis=1)
            trc_pts, trc_dof = np.where(np.abs(traction) > self._EPS_MIN)
            bc_table[trc_dof, trc_pts] = LOAD
        else:
            traction = np.zeros((self._mesh.n_point, 3))
        # Applied force
        if "applied_force" in self._mesh.bc:
            appl_force = np.float32(self._mesh.bc["applied_force"])
            if appl_force.shape[1] == 2:
                appl_force = np.insert(appl_force, 2, 0.0, axis=1)
            af_pts, af_dof = np.where(np.abs(appl_force) > self._EPS_MIN)
            bc_table[af_dof, af_pts] = FORCE
        else:
            appl_force = np.zeros((self._mesh.n_point, 3))

        # B.C. table
        f.write(
            "<DataArray NumberOfComponents='3' type='Int8' Name='Boundary Condition' format='ascii'>\n"
        )
        for ipnt in range(self._mesh.n_point):
            f.write(
                "{} {} {}\n".format(
                    bc_table[0, ipnt], bc_table[1, ipnt], bc_table[2, ipnt]
                )
            )
        f.write("</DataArray>\n")

        # Prescribed displacement
        prescribed_disp = np.zeros((3, self._mesh.n_point))
        if self._mesh.bc["n_disp"] > 0:
            prescribed_disp[disp_dof, disp_pts] = self._mesh.bc["displacement"]
        f.write(
            "<DataArray NumberOfComponents='3' type='Float32' Name='Prescribed Displacement' format='ascii'>\n"
        )
        for ipnt in range(self._mesh.n_point):
            f.write(
                "{:e} {:e} {:e}\n".format(
                    prescribed_disp[0, ipnt],
                    prescribed_disp[1, ipnt],
                    prescribed_disp[2, ipnt],
                )
            )
        f.write("</DataArray>\n")

        # Prescribed traction
        f.write(
            "<DataArray NumberOfComponents='3' type='Float32' Name='Prescribed Traction' format='ascii'>\n"
        )
        for ipnt in range(self._mesh.n_point):
            f.write(
                "{:e} {:e} {:e}\n".format(
                    traction[ipnt, 0], traction[ipnt, 1], traction[ipnt, 2]
                )
            )
        f.write("</DataArray>\n")

        # Applied force
        f.write(
            "<DataArray NumberOfComponents='3' type='Float32' Name='Applied Force' format='ascii'>\n"
        )
        for ipnt in range(self._mesh.n_point):
            f.write(
                "{:e} {:e} {:e}\n".format(
                    appl_force[ipnt, 0], appl_force[ipnt, 1], appl_force[ipnt, 2]
                )
            )
        f.write("</DataArray>\n")

        # MPC
        mpc_table = np.zeros((3, self._mesh.n_point), dtype="int32")
        mpc_ratio = np.zeros((3, self._mesh.n_point))
        if self._mesh.mpc["nmpcpt"] > 0:
            idx_mpc = np.arange(1, self._mesh.mpc["nmpcpt"] + 1)
            # Slave
            slv_pts, slv_dof = np.divmod(
                np.array(self._mesh.mpc["slave"]), self._mesh.n_dof
            )
            mpc_table[slv_dof, slv_pts] = idx_mpc
            mpc_ratio[slv_dof, slv_pts] = self._mesh.mpc["ratio"]
            # Master
            mst_pts, mst_dof = np.divmod(
                np.array(self._mesh.mpc["master"]), self._mesh.n_dof
            )
            mpc_table[mst_dof, mst_pts] = idx_mpc

        # MPC table
        f.write(
            "<DataArray NumberOfComponents='3' type='UInt8' Name='Multi-Point Constraints' format='ascii'>\n"
        )
        for ipnt in range(self._mesh.n_point):
            f.write(
                "{} {} {}\n".format(
                    mpc_table[0, ipnt], mpc_table[1, ipnt], mpc_table[2, ipnt]
                )
            )
        f.write("</DataArray>\n")

        # MPC ratio
        f.write(
            "<DataArray NumberOfComponents='3' type='Float32' Name='MPC Ratio' format='ascii'>\n"
        )
        for ipnt in range(self._mesh.n_point):
            f.write(
                "{:e} {:e} {:e}\n".format(
                    mpc_ratio[0, ipnt], mpc_ratio[1, ipnt], mpc_ratio[2, ipnt]
                )
            )
        f.write("</DataArray>\n")

    def _write_pnt_value(self, f, key: str):

        val = self._values["point"][key]
        val = val.astype("float32")
        val[np.abs(val) < self._EPS_MIN] = 0.0
        if val.ndim == 1:
            val = val[np.newaxis, :]

        f.write(
            "<DataArray NumberOfComponents='"
            + str(val.shape[0])
            + "' type='Float32' Name='"
            + key
            + "' format='ascii'>\n"
        )

        for ipnt in range(self._mesh.n_point):
            f.write(" ".join(map(str, val[:, ipnt])) + "\n")

        f.write("</DataArray>\n")

    def _write_elm_value(self, f, key: str):

        val = self._values["element"][key]
        val = val.astype("float32")
        val[np.abs(val) < self._EPS_MIN] = 0.0
        if val.ndim == 1:
            val = val[:, np.newaxis]

        f.write(
            "<DataArray NumberOfComponents='"
            + str(val.shape[1])
            + "' type='Float32' Name='"
            + key
            + "' format='ascii'>\n"
        )

        for elm in range(self._mesh.n_element):
            f.write(" ".join(map(str, val[elm])) + "\n")

        f.write("</DataArray>\n")

    def write(self, file_path: str, **kwargs) -> None:
        """
        Write .vtu file

        Parameters
        ----------
        file_path : str
            Output file path
        output_bc : bool
            Output B.C. or not
        point : list
            Output list of node values
        element : list
            Output list of element values
        """

        output_bc = kwargs["output_bc"] if "output_bc" in kwargs else True

        out_pnt_list = []
        if "point" in kwargs:
            for key in kwargs["point"]:
                if key in self._values["point"].keys():
                    out_pnt_list.append(key)
        else:
            out_pnt_list = self._values["point"].keys()

        out_elm_list = []
        if "element" in kwargs:
            for key in kwargs["element"]:
                if key in self._values["element"].keys():
                    out_elm_list.append(key)
        else:
            out_elm_list = self._values["element"].keys()

        with open(file_path, mode="w", newline="\n") as f:

            # Header
            f.write("<?xml version='1.0' encoding='utf-8'?>\n")
            f.write(
                "<VTKFile byte_order='LittleEndian' type='UnstructuredGrid' version='0.1' xmlns='VTK'>\n"
            )
            f.write("<UnstructuredGrid>\n")
            f.write(
                "<Piece NumberOfCells='"
                + str(self._mesh.n_element)
                + "' NumberOfPoints='"
                + str(self._mesh.n_point)
                + "'>\n"
            )

            # Coordinates
            coords = np.float32(self._mesh.coords)
            if coords.shape[0] == 2:
                coords = np.insert(coords, 2, 0.0, axis=0)
            f.write("<Points>\n")
            f.write(
                "<DataArray NumberOfComponents='3' type='Float32' Name='Position' format='ascii'>\n"
            )
            for ipnt in range(self._mesh.n_point):
                f.write(
                    "{:e} {:e} {:e}\n".format(
                        coords[0, ipnt], coords[1, ipnt], coords[2, ipnt]
                    )
                )
            f.write("</DataArray>\n")
            f.write("</Points>\n")

            # Begin cell value
            f.write("<Cells>\n")

            # Connectivity
            f.write("<DataArray type='Int32' Name='connectivity' format='ascii'>\n")
            for ielm, lnd in enumerate(self._mesh.connectivity):
                lnd = self._rearrange_node_numbers(lnd, self._mesh.element_name[ielm])
                for l in lnd:
                    f.write(str(l) + " ")
                f.write("\n")
            f.write("</DataArray>\n")

            # Cumulative number of nodes
            f.write("<DataArray type='Int32' Name='offsets' format='ascii'>\n")
            n_totpnt = 0
            for lnd in self._mesh.connectivity:
                n_totpnt += len(lnd)
                f.write(str(n_totpnt) + "\n")
            f.write("</DataArray>\n")

            # Cell type
            f.write("<DataArray type='UInt8' Name='types' format='ascii'>\n")
            for ename in self._mesh.element_name:
                f.write(self._get_cell_type(ename) + "\n")
            f.write("</DataArray>\n")

            # End cell data
            f.write("</Cells>\n")

            # Write point data
            if output_bc or len(out_pnt_list) > 0:

                # Start point data
                f.write("<PointData>\n")

                # Write B.C.
                if output_bc:
                    self._write_bc(f)

                # Node value
                for key in out_pnt_list:
                    self._write_pnt_value(f, key)

                # End point data
                f.write("</PointData>\n")

            # Start cell data
            f.write("<CellData>\n")

            # Grain number
            f.write(
                "<DataArray NumberOfComponents='1' type='UInt8' Name='Cryst' format='ascii'>\n"
            )
            for ielm in range(self._mesh.n_element):
                f.write("{}\n".format(self._mesh.grain_numbers[ielm]))
            f.write("</DataArray>\n")

            # Material number
            f.write(
                "<DataArray NumberOfComponents='1' type='UInt8' Name='Mater' format='ascii'>\n"
            )
            for ielm in range(self._mesh.n_element):
                f.write("{}\n".format(self._mesh.material_numbers[ielm]))
            f.write("</DataArray>\n")

            # Crystal orientation
            f.write(
                "<DataArray NumberOfComponents='3' type='Float32' Name='Crystal Orientation' format='ascii'>\n"
            )
            for ielm in range(self._mesh.n_element):
                f.write(
                    "{:e} {:e} {:e}\n".format(
                        self._mesh.crystal_orientation[ielm, 0],
                        self._mesh.crystal_orientation[ielm, 1],
                        self._mesh.crystal_orientation[ielm, 2],
                    )
                )
            f.write("</DataArray>\n")

            # Element values
            for key in out_elm_list:
                self._write_elm_value(f, key)

            # End cell data
            f.write("</CellData>\n")

            # Footer
            f.write("</Piece>\n")
            f.write("</UnstructuredGrid>\n")
            f.write("</VTKFile>\n")
            f.close()
