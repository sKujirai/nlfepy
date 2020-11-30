import sys
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .viewer_base import ViewerBase


class Viewer3d(ViewerBase):
    """
    Viewer class for 3D stuructures inheriting class: ViewerBase
    """

    def __init__(self) -> None:
        super().__init__()

    def _set_window(self, coords: np.ndarray, **kwargs) -> None:
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
        zlim : array-like
            Range of z-axis
        """

        plt.close()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        # self._ax.axis('off')
        self._ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self._ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self._ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self._ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self._ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self._ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        self._ax.axes.xaxis.set_ticks([])
        self._ax.axes.yaxis.set_ticks([])
        self._ax.axes.zaxis.set_ticks([])

        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')

        if 'xlim' in kwargs:
            self._ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        else:
            self._ax.set_xlim(np.min(coords[0]), np.max(coords[0]))
        if 'ylim' in kwargs:
            self._ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
        else:
            self._ax.set_ylim(np.min(coords[1]), np.max(coords[1]))
        if 'zlim' in kwargs:
            self._ax.set_zlim(kwargs['zlim'][0], kwargs['zlim'][1])
        else:
            self._ax.set_zlim(np.min(coords[2]), np.max(coords[2]))

    def _set_bc_info(self, mesh) -> None:
        """
        Plot boundary conditions

        Parameters
        ----------
        mesh :
            Mesh class (See mesh.py)
        """

        rr = 0.2
        eps_crit = 1.e-10
        eps_min = 1.e-30

        disp_pts, disp_dof = np.divmod(mesh.bc['idx_disp'], mesh.n_dof)
        idx_disp_x = np.where(disp_dof == 0)
        idx_disp_y = np.where(disp_dof == 1)
        idx_disp_z = np.where(disp_dof == 2)
        fix_pts, fix_dof = np.divmod(mesh.bc['idx_fix'], mesh.n_dof)
        fix_x = fix_pts[fix_dof == 0]
        fix_y = fix_pts[fix_dof == 1]
        fix_z = fix_pts[fix_dof == 2]
        fix_x = np.setdiff1d(fix_x, disp_pts[idx_disp_x])
        fix_y = np.setdiff1d(fix_y, disp_pts[idx_disp_y])
        fix_z = np.setdiff1d(fix_z, disp_pts[idx_disp_z])
        fix_xyz = np.intersect1d(fix_x, np.intersect1d(fix_y, fix_z))
        fix_xy = np.setdiff1d(np.intersect1d(fix_x, fix_y), fix_xyz)
        fix_yz = np.setdiff1d(np.intersect1d(fix_y, fix_z), fix_xyz)
        fix_zx = np.setdiff1d(np.intersect1d(fix_z, fix_x), fix_xyz)
        fix_x = np.setdiff1d(np.setdiff1d(np.setdiff1d(fix_x, fix_xy), fix_zx), fix_xyz)
        fix_y = np.setdiff1d(np.setdiff1d(np.setdiff1d(fix_y, fix_yz), fix_xy), fix_xyz)
        fix_z = np.setdiff1d(np.setdiff1d(np.setdiff1d(fix_z, fix_zx), fix_yz), fix_xyz)

        ll = max(np.max(mesh.coords, axis=1)) * rr

        # Fix points
        self._ax.scatter3D(
            mesh.coords[0, fix_xyz],
            mesh.coords[1, fix_xyz],
            mesh.coords[2, fix_xyz],
            color=r'#e41a1c',
            label='Completely fixed',
        )

        if len(fix_xy) > 0:
            self._ax.scatter3D(
                mesh.coords[0, fix_xy],
                mesh.coords[1, fix_xy],
                mesh.coords[2, fix_xy],
                color=r'#ff7f00',
                label='Fix x-y',
            )

        if len(fix_yz) > 0:
            self._ax.scatter3D(
                mesh.coords[0, fix_yz],
                mesh.coords[1, fix_yz],
                mesh.coords[2, fix_yz],
                color=r'#a65628',
                label='Fix y-z',
            )

        if len(fix_zx) > 0:
            self._ax.scatter3D(
                mesh.coords[0, fix_zx],
                mesh.coords[1, fix_zx],
                mesh.coords[2, fix_zx],
                color=r'#f781bf',
                label='Fix z-x',
            )

        if len(fix_x) > 0:
            self._ax.scatter3D(
                mesh.coords[0, fix_x],
                mesh.coords[1, fix_x],
                mesh.coords[2, fix_x],
                color=r'#377eb8',
                label='Fix x',
            )

        if len(fix_y) > 0:
            self._ax.scatter3D(
                mesh.coords[0, fix_y],
                mesh.coords[1, fix_y],
                mesh.coords[2, fix_y],
                color=r'#4daf4a',
                label='Fix y',
            )

        if len(fix_z) > 0:
            self._ax.scatter3D(
                mesh.coords[0, fix_z],
                mesh.coords[1, fix_z],
                mesh.coords[2, fix_z],
                color=r'#984ea3',
                label='Fix z',
            )

        # Prescribed displacement
        is_plot_pd = False

        if len(idx_disp_x[0]) > 0:
            self._ax.quiver(
                mesh.coords[0, disp_pts[idx_disp_x]],
                mesh.coords[1, disp_pts[idx_disp_x]],
                mesh.coords[2, disp_pts[idx_disp_x]],
                np.sign(mesh.bc['displacement'][idx_disp_x]),
                0.,
                0.,
                length=ll,
                color='r',
                label='Prescribed displacement',
            )
            is_plot_pd = True

        if len(idx_disp_y[0]) > 0:
            label_disp_y = None if is_plot_pd else 'Prescribed displacement'
            self._ax.quiver(
                mesh.coords[0, disp_pts[idx_disp_y]],
                mesh.coords[1, disp_pts[idx_disp_y]],
                mesh.coords[2, disp_pts[idx_disp_y]],
                0.,
                np.sign(mesh.bc['displacement'][idx_disp_y]),
                0.,
                length=ll,
                color='r',
                label=label_disp_y,
            )
            is_plot_pd = True

        if len(idx_disp_z[0]) > 0:
            label_disp_z = None if is_plot_pd else 'Prescribed displacement'
            self._ax.quiver(
                mesh.coords[0, disp_pts[idx_disp_z]],
                mesh.coords[1, disp_pts[idx_disp_z]],
                mesh.coords[2, disp_pts[idx_disp_z]],
                0.,
                0.,
                np.sign(mesh.bc['displacement'][idx_disp_z]),
                length=ll,
                color='r',
                label=label_disp_z,
            )

        # Traction
        if 'traction' in mesh.bc:
            Traction = mesh.bc['traction']
            max_trc = np.max(np.abs(Traction))
            if max_trc > eps_min:
                trc_crit = max_trc * eps_crit
                pnt_trc, dof_trc = np.where(np.abs(Traction) > trc_crit)
                idx_trc_x = np.where(dof_trc == 0)
                idx_trc_y = np.where(dof_trc == 1)
                idx_trc_z = np.where(dof_trc == 2)

                is_plot_trc = False

                if len(idx_trc_x[0]) > 0:
                    self._ax.quiver(
                        mesh.coords[0, pnt_trc[idx_trc_x]],
                        mesh.coords[1, pnt_trc[idx_trc_x]],
                        mesh.coords[2, pnt_trc[idx_trc_x]],
                        np.sign(Traction[pnt_trc[idx_trc_x], 0]),
                        0.,
                        0.,
                        length=ll,
                        color='g',
                        label='Tractopm',
                    )
                    is_plot_trc = True

                if len(idx_trc_y[0]) > 0:
                    label_trc_y = None if is_plot_trc else 'Traction'
                    self._ax.quiver(
                        mesh.coords[0, pnt_trc[idx_trc_y]],
                        mesh.coords[1, pnt_trc[idx_trc_y]],
                        mesh.coords[2, pnt_trc[idx_trc_y]],
                        0.,
                        np.sign(Traction[pnt_trc[idx_trc_y], 1]),
                        0.,
                        length=ll,
                        color='g',
                        label=label_trc_y,
                    )
                    is_plot_trc = True

                if len(idx_trc_z[0]) > 0:
                    label_trc_z = None if is_plot_trc else 'Traction'
                    self._ax.quiver(
                        mesh.coords[0, pnt_trc[idx_trc_z]],
                        mesh.coords[1, pnt_trc[idx_trc_z]],
                        mesh.coords[2, pnt_trc[idx_trc_z]],
                        0.,
                        0.,
                        np.sign(Traction[pnt_trc[idx_trc_z], 2]),
                        length=ll,
                        color='g',
                        label=label_trc_z,
                    )
                    is_plot_trc = True

        # Applied force
        if 'applied_force' in mesh.bc:
            ApplForce = mesh.bc['applied_force']
            max_af = np.max(np.abs(ApplForce))
            if max_af > eps_min:
                af_crit = max_af * eps_crit
                pnt_af, dof_af = np.where(np.abs(ApplForce) > af_crit)
                idx_af_x = np.where(dof_af == 0)
                idx_af_y = np.where(dof_af == 1)
                idx_af_z = np.where(dof_af == 2)

                is_plot_af = False

                if len(idx_af_x[0]) > 0:
                    self._ax.quiver(
                        mesh.coords[0, pnt_af[idx_af_x]],
                        mesh.coords[1, pnt_af[idx_af_x]],
                        mesh.coords[2, pnt_af[idx_af_x]],
                        np.sign(ApplForce[pnt_af[idx_af_x], 0]),
                        0.,
                        0.,
                        length=ll,
                        color='b',
                        label='Applied force',
                    )
                    is_plot_af = True

                if len(idx_af_y[0]) > 0:
                    label_af_y = None if is_plot_af else 'Applied force'
                    self._ax.quiver(
                        mesh.coords[0, pnt_af[idx_af_y]],
                        mesh.coords[1, pnt_af[idx_af_y]],
                        mesh.coords[2, pnt_af[idx_af_y]],
                        0.,
                        np.sign(ApplForce[pnt_af[idx_af_y], 1]),
                        0.,
                        length=ll,
                        color='b',
                        label=label_af_y,
                    )
                    is_plot_af = True

                if len(idx_af_z[0]) > 0:
                    label_af_z = None if is_plot_af else 'Applied force'
                    self._ax.quiver(
                        mesh.coords[0, pnt_af[idx_af_z]],
                        mesh.coords[1, pnt_af[idx_af_z]],
                        mesh.coords[2, pnt_af[idx_af_z]],
                        0.,
                        0.,
                        np.sign(ApplForce[pnt_af[idx_af_z], 2]),
                        length=ll,
                        color='b',
                        label=label_af_z,
                    )
                    is_plot_af = True

        plt.legend()

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
        zlim : array-like
            Range of z-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        alpha : float
            Transparency
        """

        self._set_window(mesh.coords, **kwargs)

        verts = []
        vals = []
        if val is not None:
            val = np.array(val)
            if val.shape[0] != mesh.n_element:
                logger = getLogger('viewer3d')
                logger.error('Value arrray has invalid size. Size of values: {}, Number of elements: {}'.format(val.shape[0], mesh.n_element))
                sys.exit(1)

        for ielm in range(mesh.n_element):
            for idx_nd in mesh.idx_face('vol', elm=ielm):
                cod = mesh.coords[:, np.array(mesh.connectivity[ielm])[idx_nd]].T
                verts.append(cod)
                if val is not None:
                    vals.append(val[ielm])

        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'k'
        if 'lw' not in kwargs:
            kwargs['lw'] = 1
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'rainbow'
        if 'facecolors' not in kwargs:
            kwargs['facecolors'] = 'orange'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.25

        self._pcm = Poly3DCollection(
            verts,
            **kwargs
        )

        if len(vals) > 0:
            self._pcm.set_array(np.array(vals))

        self._ax.add_collection3d(self._pcm)

        self.show_cbar = True

    def plot_bc(self, mesh, **kwargs) -> None:
        """
        Plot boundary conditions

        Parameters
        ----------
        mesh :
            Mesh class
        """

        self.plot(mesh=mesh, **kwargs)

        self._set_bc_info(mesh)

        self.show_cbar = False

        self.show_cbar = False

    def contour(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        """
        Contour plot

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
        zlim : array-like
            Range of z-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        alpha : float
            Transparency
        """

        self._set_window(mesh.coords, **kwargs)

        val = np.array(val)
        if val.shape[0] != mesh.n_point:
            logger = getLogger('viewer3d')
            logger.error('Value arrray has invalid size. Size of values: {}, Number of elements: {}'.format(val.shape[0], mesh.n_point))
            sys.exit(1)

        self.show_cbar = True

        # Not implemented
        raise NotImplementedError()
