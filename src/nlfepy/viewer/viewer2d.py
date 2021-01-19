import sys
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
from .viewer_base import ViewerBase


class Viewer2d(ViewerBase):
    """
    Viewer class for 2D stuructures inheriting class: ViewerBase
    """

    def __init__(self) -> None:
        super().__init__()

    def _set_window(self, ax, coords: np.ndarray, **kwargs):
        """
        Set window

        Parameters
        ----------
        coords : ndarray
            Coordinates [n_dof, n_point]
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis

        Returns
        -------
        ax :
            Axis
        """

        # ax.axis('off')
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_aspect("equal")

        if "show_axis_label" in kwargs and kwargs["show_axis_label"]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        ratio_margin = 0.1
        specimen_size = np.max(coords, axis=1)

        if "xlim" in kwargs:
            ax.set_xlim(kwargs["xlim"][0], kwargs["xlim"][1])
        else:
            ax.set_xlim(
                np.min(coords[0]) - specimen_size[0] * ratio_margin,
                np.max(coords[0]) + specimen_size[0] * ratio_margin,
            )
        if "ylim" in kwargs:
            ax.set_ylim(kwargs["ylim"][0], kwargs["ylim"][1])
        else:
            ax.set_ylim(
                np.min(coords[1]) - specimen_size[1] * ratio_margin,
                np.max(coords[1]) + specimen_size[1] * ratio_margin,
            )

        return ax

    def _set_bc_info(self, ax, mesh):
        """
        Plot boundary conditions

        Parameters
        ----------
        mesh :
            Mesh class (See mesh.py)

        Returns
        -------
        ax :
            Axis
        """

        rr = 0.2
        eps_crit = 1.0e-10
        eps_min = 1.0e-30

        disp_pts, disp_dof = np.divmod(mesh.bc["idx_disp"], mesh.n_dof)
        idx_disp_x = np.where(disp_dof == 0)
        idx_disp_y = np.where(disp_dof == 1)
        fix_pts, fix_dof = np.divmod(mesh.bc["idx_fix"], mesh.n_dof)
        fix_x = fix_pts[fix_dof == 0]
        fix_y = fix_pts[fix_dof == 1]
        fix_x = np.setdiff1d(fix_x, disp_pts[idx_disp_x])
        fix_y = np.setdiff1d(fix_y, disp_pts[idx_disp_y])
        fix_xy = np.intersect1d(fix_x, fix_y)
        fix_x = np.setdiff1d(fix_x, fix_xy)
        fix_y = np.setdiff1d(fix_y, fix_xy)

        ll = max(np.max(mesh.coords[: mesh.n_dof], axis=1)) * rr

        # Fix points
        ax.scatter(
            mesh.coords[0, fix_xy],
            mesh.coords[1, fix_xy],
            color="r",
            label="Completely fixed",
        )

        if len(fix_x) > 0:
            ax.scatter(
                mesh.coords[0, fix_x],
                mesh.coords[1, fix_x],
                color="b",
                label="Fixed in x direction",
            )

        if len(fix_y) > 0:
            ax.scatter(
                mesh.coords[0, fix_y],
                mesh.coords[1, fix_y],
                color="g",
                label="Fixed in y direction",
            )

        # Prescribed displacement
        ratio_disp = (
            ll / abs(np.max(mesh.bc["displacement"])) * rr
            if len(mesh.bc["idx_disp"]) > 0
            else None
        )

        if len(idx_disp_x[0]) > 0:
            ax.quiver(
                mesh.coords[0, disp_pts[idx_disp_x]],
                mesh.coords[1, disp_pts[idx_disp_x]],
                mesh.bc["displacement"][idx_disp_x] * ratio_disp,
                0.0,
                color="r",
                label="Prescribed displacement",
            )

        if len(idx_disp_y[0]) > 0:
            label_disp_y = None if len(idx_disp_x[0]) > 0 else "Prescribed displacement"
            ax.quiver(
                mesh.coords[0, disp_pts[idx_disp_y]],
                mesh.coords[1, disp_pts[idx_disp_y]],
                0.0,
                mesh.bc["displacement"][idx_disp_y] * ratio_disp,
                color="r",
                label=label_disp_y,
            )

        # Traction
        if "traction" in mesh.bc:
            Traction = mesh.bc["traction"]
            max_trc = np.max(np.abs(Traction))
            if max_trc > eps_min:
                trc_crit = max_trc * eps_crit
                pnt_trc, dof_trc = np.where(np.abs(Traction) > trc_crit)
                idx_trc_x = np.where(dof_trc == 0)
                idx_trc_y = np.where(dof_trc == 1)
                ratio_trc = ll / max_trc * rr

                if len(idx_trc_x[0]) > 0:
                    ax.quiver(
                        mesh.coords[0, pnt_trc[idx_trc_x]],
                        mesh.coords[1, pnt_trc[idx_trc_x]],
                        Traction[pnt_trc[idx_trc_x], 0] * ratio_trc,
                        0.0,
                        color="g",
                        label="Traction",
                    )

                if len(idx_trc_y[0]) > 0:
                    label_trc_y = None if len(idx_trc_x[0]) > 0 else "Traction"
                    ax.quiver(
                        mesh.coords[0, pnt_trc[idx_trc_y]],
                        mesh.coords[1, pnt_trc[idx_trc_y]],
                        0.0,
                        Traction[pnt_trc[idx_trc_y], 1] * ratio_trc,
                        color="g",
                        label=label_trc_y,
                    )

        # Applied force
        if "applied_force" in mesh.bc:
            ApplForce = mesh.bc["applied_force"]
            max_af = np.max(np.abs(ApplForce))
            if max_af > eps_min:
                af_crit = max_af * eps_crit
                pnt_af, dof_af = np.where(np.abs(ApplForce) > af_crit)
                idx_af_x = np.where(dof_af == 0)
                idx_af_y = np.where(dof_af == 1)
                ratio_af = ll / max_af * rr

                if len(idx_af_x[0]) > 0:
                    ax.quiver(
                        mesh.coords[0, pnt_af[idx_af_x]],
                        mesh.coords[1, pnt_af[idx_af_x]],
                        ApplForce[pnt_af[idx_af_x], 0] * ratio_af,
                        0.0,
                        color="b",
                        label="Applied force",
                    )

                if len(idx_af_y[0]) > 0:
                    label_trc_y = None if len(idx_af_y[0]) > 0 else "Applied force"
                    ax.quiver(
                        mesh.coords[0, pnt_af[idx_af_y]],
                        mesh.coords[1, pnt_af[idx_af_y]],
                        0.0,
                        ApplForce[pnt_af[idx_af_y], 1] * ratio_af,
                        color="b",
                        label=label_trc_y,
                    )

        ax.legend()

        return ax

    def _ax_plot(self, *, ax, mesh, val: np.ndarray = None, **kwargs):
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width

        Returns
        -------
        ax :
            Axis
        pcm :
            PolyCollection
        """

        ax = self._set_window(ax, mesh.coords, **kwargs)

        vertices = mesh.coords[:2, :].T[np.asarray(mesh.connectivity)]

        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "k"
        if "lw" not in kwargs:
            kwargs["lw"] = 1
        if "cmap" not in kwargs:
            kwargs["cmap"] = "rainbow"
        self._delete_plt_unnecessary_keys(kwargs)

        if val is None:
            pcm = PolyCollection(vertices, facecolor="None", **kwargs)
        else:
            pcm = PolyCollection(vertices, **kwargs)
            value = np.array(val)
            if value.shape[0] != mesh.n_element:
                logger = getLogger("viewer2d")
                logger.error(
                    "Value arrray has invalid size. Size of values: {}, Number of elements: {}".format(
                        value.shape[0], mesh.n_element
                    )
                )
                sys.exit(1)
            pcm.set_array(value)
        ax.add_collection(pcm)

        return ax, pcm

    def _ax_contour(self, *, ax, mesh, val: np.ndarray, **kwargs):
        """
        Contour plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each nodes [n_point] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width

        Returns
        -------
        ax :
            Axis
        pcm :
            PolyCollection
        """

        ax = self._set_window(ax, mesh.coords, **kwargs)

        self._delete_plt_unnecessary_keys(kwargs)

        logger = getLogger("viewer2d")

        value = np.array(val)
        if value.shape[0] != mesh.n_point:
            logger.error(
                "Value arrray has invalid size. Size of values: {}, Number of total nodes: {}".format(
                    value.shape[0], mesh.n_point
                )
            )
            sys.exit(1)

        # TRI6, QUAD -> TRI3
        connectivity_tri3 = []
        for ielm, lnod in enumerate(mesh.connectivity):
            element_shape = mesh.element_shape[ielm]
            if element_shape == "TRI":
                connectivity_tri3.append([lnod[0], lnod[1], lnod[2]])
            elif element_shape == "QUAD":
                connectivity_tri3.append([lnod[0], lnod[1], lnod[2]])
                connectivity_tri3.append([lnod[0], lnod[2], lnod[3]])

        triang = mtri.Triangulation(
            mesh.coords[0, :], mesh.coords[1, :], connectivity_tri3
        )
        pcm = ax.tricontourf(triang, value, **kwargs)

        return ax, pcm

    def _ax_scatter(self, *, ax, mesh, val: np.ndarray, **kwargs):
        """
        Contour plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each nodes [n_point] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width

        Returns
        -------
        ax :
            Axis
        pcm :
            PolyCollection
        """

        ax = self._set_window(ax, mesh.coords, **kwargs)

        self._delete_plt_unnecessary_keys(kwargs)

        ax.scatter(mesh.coords[0, :], mesh.coords[1, :], c=val, **kwargs)

        pcm = ax.get_children()[0]

        return ax, pcm

    def plot(self, *, mesh, val: np.ndarray = None, **kwargs) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        """

        kwargs["projection"] = None

        super().plot(mesh=mesh, val=val, **kwargs)

    def contour(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        """
        Contour plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each nodes [n_point] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        """

        kwargs["projection"] = None

        super().contour(mesh=mesh, val=val, **kwargs)

    def scatter(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        """
        Scatter plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each nodes [n_point] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        """

        kwargs["projection"] = None

        super().scatter(mesh=mesh, val=val, **kwargs)

    def multi_plot(self, mesh, vlist, **kwargs) -> None:
        """
        Plot multiple figures

        Parameters
        ----------
        mesh :
            Mesh class
        vlist: list
            List of configurations.
            'val': Value to plot
            'plot': Plot mode ('fill' or 'scatter')
            'figname': Figure name
        """

        kwargs["projection"] = None

        super().multi_plot(mesh, vlist, **kwargs)
