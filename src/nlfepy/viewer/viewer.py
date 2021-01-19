import os
import sys
import glob
import re
import numpy as np
from logging import getLogger
from .viewer_base import ViewerBase
from .viewer2d import Viewer2d
from .viewer3d import Viewer3d
from ..io.vtu_reader import VtuReader
from ..mesh.mesh import Mesh


class Viewer:
    """
    Viewer interface
    """

    def __init__(self, projection: str = None) -> None:

        self._viewer: ViewerBase
        if projection == "3d":
            self._viewer = Viewer3d()
        else:
            self._viewer = Viewer2d()

    def plot(self, *, mesh, val: np.ndarray = None, **kwargs) -> None:
        self._viewer.plot(mesh=mesh, val=val, **kwargs)

    def plot_bc(self, mesh, **kwargs) -> None:
        self._viewer.plot_bc(mesh=mesh, **kwargs)

    def contour(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        self._viewer.contour(mesh=mesh, val=val, **kwargs)

    def scatter(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        self._viewer.scatter(mesh=mesh, val=val, **kwargs)

    def multi_plot(self, file, cnfs: list, **kwargs) -> None:
        """
        Plot multile figures
        """

        mesh, val_list = self._get_vtk_values(file, cnfs)

        self._viewer.multi_plot(mesh=mesh, vlist=val_list, **kwargs)

    def _get_vtk_values(self, file, cnfs: list):

        reader = VtuReader()
        reader.read(file)
        mesh = Mesh()
        mesh.set_mesh_data(mesh=reader.mesh, bc=reader.bc, mpc=reader.mpc)

        val_list = []
        for cnf in cnfs:
            if "val" not in cnf.keys() or cnf["val"] is None:
                val_list.append(
                    {
                        "val": None,
                        "figname": "wireframe",
                        "plot": "fill",
                    }
                )
            else:
                sys = cnf["sys"] if "sys" in cnf.keys() else None
                plot_mode = cnf["plot"] if "plot" in cnf.keys() else "fill"
                if plot_mode in ["contour", "scatter"]:
                    val = reader.get_point_value(cnf["val"], systems=sys).astype(np.float)
                else:
                    val = reader.get_elm_value(cnf["val"], systems=sys).astype(np.float)
                if sys is None:
                    sys = [i for i in range(val.shape[1])]
                for i, isys in enumerate(sys):
                    val_list.append(
                        {
                            "val": val[:, i],
                            "figname": cnf["val"] + " " + str(isys),
                            "plot": plot_mode,
                        }
                    )

        return mesh, val_list

    def get_frame_imgs(self, *, fig_cnfs: list, vtk_cnf: dict = {}, **kwargs) -> list:
        """
        Get image of each frame
        """

        vtk_num_list = self._get_vtk_num_list(vtk_cnf=vtk_cnf)

        imgs = []
        for inum in vtk_num_list:
            self.multi_plot(
                os.path.join(
                    vtk_cnf["dir"], vtk_cnf["header"] + str(inum) + vtk_cnf["ext"]
                ),
                fig_cnfs,
                **kwargs
            )
            im = self._viewer.get_fig_array()
            # self.save('tmp.png', dpi=300)
            # im = np.array(Image.open('tmp.png'))
            imgs.append(im)

        return imgs

    def save_figs(self, *, fig_cnfs: list, vtk_cnf: dict = {}, **kwargs) -> None:
        """
        Save images of each frame
        """

        vtk_num_list = self._get_vtk_num_list(vtk_cnf=vtk_cnf)

        # Set output file name
        if "out_dir" not in vtk_cnf:
            vtk_cnf["out_dir"] = vtk_cnf["dir"]
        if "out_header" not in vtk_cnf:
            vtk_cnf["out_header"] = vtk_cnf["header"]
        if "out_ext" not in vtk_cnf:
            vtk_cnf["out_ext"] = ".png"

        save_cnf = {}
        if "dpi" in kwargs:
            save_cnf["dpi"] = kwargs["dpi"]
            del kwargs["dpi"]
        if "transparent" in kwargs:
            save_cnf["transparent"] = kwargs["transparent"]
            del kwargs["transparent"]

        if not os.path.exists(vtk_cnf["out_dir"]):
            logger = getLogger("viewer interface")
            try:
                os.mkdir(vtk_cnf["out_dir"])
            except PermissionError:
                logger.error("Permission error")
            except Exception:
                logger.error("Failed to create output directory")
                sys.exit(1)

        for inum in vtk_num_list:
            vtk_path = os.path.join(
                vtk_cnf["dir"], vtk_cnf["header"] + str(inum) + vtk_cnf["ext"]
            )
            mesh, val_list = self._get_vtk_values(vtk_path, fig_cnfs)
            for vl in val_list:
                if vl["plot"] == "contour":
                    self.contour(
                        mesh=mesh, val=vl["val"], title=vl["figname"], **kwargs
                    )
                elif vl["plot"] == "scatter":
                    self.scatter(
                        mesh=mesh, val=vl["val"], title=vl["figname"], **kwargs
                    )
                else:
                    self.plot(mesh=mesh, val=vl["val"], title=vl["figname"], **kwargs)
                self.save(
                    os.path.join(
                        vtk_cnf["out_dir"],
                        vtk_cnf["out_header"]
                        + vl["figname"]
                        + "_"
                        + str(inum)
                        + vtk_cnf["out_ext"],
                    ),
                    **save_cnf
                )

    def _get_vtk_num_list(self, *, vtk_cnf: dict = {}):

        # Get VTK file list
        if "dir" not in vtk_cnf:
            vtk_cnf["dir"] = r"./out"
        if "header" not in vtk_cnf:
            vtk_cnf["header"] = r"result_"
        if "ext" not in vtk_cnf:
            vtk_cnf["ext"] = r".vtu"

        vtk_list = glob.glob(
            os.path.join(vtk_cnf["dir"], vtk_cnf["header"]) + "*" + vtk_cnf["ext"]
        )
        # vtk_match_list = []
        vtk_num_list = []
        regex = re.compile("\d+")
        for vtk in vtk_list:
            if re.match(vtk_cnf["header"] + "[0-9]+.vtu", os.path.basename(vtk)):
                # vtk_match_list.append(vtk)
                vtk_num_list.append(int(regex.findall(vtk)[-1]))

        # vtk_list = vtk_match_list
        vtk_num_list.sort()

        return vtk_num_list

    def show(self) -> None:
        self._viewer.show()

    def save(self, file_name, **kwargs) -> None:
        self._viewer.save(file_name, **kwargs)
