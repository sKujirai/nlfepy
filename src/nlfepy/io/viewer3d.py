import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .viewer_base import ViewerBase


class Viewer3d(ViewerBase):
    """
    Viewer class for 3D stuructures inheriting class: ViewerBase
    """

    def __init__(self, *, mesh, set_mesh_info=True) -> None:

        super().__init__(mesh=mesh)

        if set_mesh_info:
            self.set()
            self.set_bc_info()

    def set_window(self, *, params: dict = {}) -> None:
        """
        Set window

        Parameters
        ----------
        params : dict
            'xlim' (array-like, Range of x-axis),
            'ylim' (array-like, Range of y-axis),
            'zlim' (array-like, Range of z-axis),
            'cmap' (string, Color map),
            'edgecolor' (string, Edge color),
            'lw' (int, Line width),
            'alpha' (float, Transparency)
        """

        plt.close()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # self.ax.axis('off')
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        self.ax.axes.xaxis.set_ticks([])
        self.ax.axes.yaxis.set_ticks([])
        self.ax.axes.zaxis.set_ticks([])

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        if 'xlim' in params:
            self.ax.set_xlim(params['xlim'][0], params['xlim'][1])
        else:
            self.ax.set_xlim(np.min(self.mesh.coords[0]), np.max(self.mesh.coords[0]))
        if 'ylim' in params:
            self.ax.set_ylim(params['ylim'][0], params['ylim'][1])
        else:
            self.ax.set_ylim(np.min(self.mesh.coords[1]), np.max(self.mesh.coords[1]))
        if 'zlim' in params:
            self.ax.set_zlim(params['zlim'][0], params['zlim'][1])
        else:
            self.ax.set_zlim(np.min(self.mesh.coords[2]), np.max(self.mesh.coords[2]))

    def set_bc_info(self) -> None:
        """
        Plot boundary conditions
        """

        rr = 0.2
        eps_crit = 1.e-10
        eps_min = 1.e-30

        disp_pts, disp_dof = np.divmod(self.mesh.bc['idx_disp'], self.mesh.n_dof)
        idx_disp_x = np.where(disp_dof == 0)
        idx_disp_y = np.where(disp_dof == 1)
        idx_disp_z = np.where(disp_dof == 2)
        fix_pts, fix_dof = np.divmod(self.mesh.bc['idx_fix'], self.mesh.n_dof)
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

        ll = max(np.max(self.mesh.coords[:self.mesh.n_dof], axis=1)) * rr

        # Fix points
        self.ax.scatter3D(
            self.mesh.coords[0, fix_xyz],
            self.mesh.coords[1, fix_xyz],
            self.mesh.coords[2, fix_xyz],
            color=r'#e41a1c',
            label='Completely fixed',
        )

        if len(fix_xy) > 0:
            self.ax.scatter3D(
                self.mesh.coords[0, fix_xy],
                self.mesh.coords[1, fix_xy],
                self.mesh.coords[2, fix_xy],
                color=r'#ff7f00',
                label='Fix x-y',
            )

        if len(fix_yz) > 0:
            self.ax.scatter3D(
                self.mesh.coords[0, fix_yz],
                self.mesh.coords[1, fix_yz],
                self.mesh.coords[2, fix_yz],
                color=r'#a65628',
                label='Fix y-z',
            )

        if len(fix_zx) > 0:
            self.ax.scatter3D(
                self.mesh.coords[0, fix_zx],
                self.mesh.coords[1, fix_zx],
                self.mesh.coords[2, fix_zx],
                color=r'#f781bf',
                label='Fix z-x',
            )

        if len(fix_x) > 0:
            self.ax.scatter3D(
                self.mesh.coords[0, fix_x],
                self.mesh.coords[1, fix_x],
                self.mesh.coords[2, fix_x],
                color=r'#377eb8',
                label='Fix x',
            )

        if len(fix_y) > 0:
            self.ax.scatter3D(
                self.mesh.coords[0, fix_y],
                self.mesh.coords[1, fix_y],
                self.mesh.coords[2, fix_y],
                color=r'#4daf4a',
                label='Fix y',
            )

        if len(fix_z) > 0:
            self.ax.scatter3D(
                self.mesh.coords[0, fix_z],
                self.mesh.coords[1, fix_z],
                self.mesh.coords[2, fix_z],
                color=r'#984ea3',
                label='Fix z',
            )

        # Prescribed displacement
        is_plot_pd = False

        if len(idx_disp_x[0]) > 0:
            self.ax.quiver(
                self.mesh.coords[0, disp_pts[idx_disp_x]],
                self.mesh.coords[1, disp_pts[idx_disp_x]],
                self.mesh.coords[2, disp_pts[idx_disp_x]],
                np.sign(self.mesh.bc['displacement'][idx_disp_x]),
                0.,
                0.,
                length=ll,
                color='r',
                label='Prescribed displacement',
            )
            is_plot_pd = True

        if len(idx_disp_y[0]) > 0:
            label_disp_y = None if is_plot_pd else 'Prescribed displacement'
            self.ax.quiver(
                self.mesh.coords[0, disp_pts[idx_disp_y]],
                self.mesh.coords[1, disp_pts[idx_disp_y]],
                self.mesh.coords[2, disp_pts[idx_disp_y]],
                0.,
                np.sign(self.mesh.bc['displacement'][idx_disp_y]),
                0.,
                length=ll,
                color='r',
                label=label_disp_y,
            )
            is_plot_pd = True

        if len(idx_disp_z[0]) > 0:
            label_disp_z = None if is_plot_pd else 'Prescribed displacement'
            self.ax.quiver(
                self.mesh.coords[0, disp_pts[idx_disp_z]],
                self.mesh.coords[1, disp_pts[idx_disp_z]],
                self.mesh.coords[2, disp_pts[idx_disp_z]],
                0.,
                np.sign(self.mesh.bc['displacement'][idx_disp_z]),
                0.,
                length=ll,
                color='r',
                label=label_disp_z,
            )

        # Traction
        if 'traction' in self.mesh.bc:
            Traction = self.mesh.bc['traction']
            max_trc = np.max(np.abs(Traction))
            if max_trc > eps_min:
                trc_crit = max_trc * eps_crit
                dof_trc, pnt_trc = np.where(np.abs(Traction) > trc_crit)
                idx_trc_x = np.where(dof_trc == 0)
                idx_trc_y = np.where(dof_trc == 1)

                is_plot_trc = False

                if len(idx_trc_x[0]) > 0:
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_trc[idx_trc_x]],
                        self.mesh.coords[1, pnt_trc[idx_trc_x]],
                        self.mesh.coords[2, pnt_trc[idx_trc_x]],
                        np.sign(Traction[0, pnt_trc[idx_trc_x]]),
                        0.,
                        0.,
                        length=ll,
                        color='g',
                        label='Tractopm',
                    )
                    is_plot_af = True

                if len(idx_trc_y[0]) > 0:
                    label_trc_y = None if is_plot_trc else 'Traction'
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_trc[idx_trc_y]],
                        self.mesh.coords[1, pnt_trc[idx_trc_y]],
                        self.mesh.coords[2, pnt_trc[idx_trc_y]],
                        0.,
                        np.sign(Traction[0, pnt_trc[idx_trc_y]]),
                        0.,
                        length=ll,
                        color='g',
                        label=label_trc_y,
                    )
                    is_plot_af = True

                if len(idx_trc_z[0]) > 0:
                    label_trc_z = None if is_plot_trc else 'Traction'
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_trc[idx_trc_z]],
                        self.mesh.coords[1, pnt_trc[idx_trc_z]],
                        self.mesh.coords[2, pnt_trc[idx_trc_z]],
                        0.,
                        np.sign(Traction[0, pnt_trc[idx_trc_z]]),
                        0.,
                        length=ll,
                        color='g',
                        label=label_trc_z,
                    )

        # Applied force
        if 'applied_force' in self.mesh.bc:
            ApplForce = self.mesh.bc['applied_force']
            max_af = np.max(np.abs(ApplForce))
            if max_af > eps_min:
                af_crit = max_af * eps_crit
                dof_af, pnt_af = np.where(np.abs(ApplForce) > af_crit)
                idx_af_x = np.where(dof_af == 0)
                idx_af_y = np.where(dof_af == 1)
                idx_af_z = np.where(dof_af == 2)

                is_plot_af = False

                if len(idx_af_x[0]) > 0:
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_af[idx_af_x]],
                        self.mesh.coords[1, pnt_af[idx_af_x]],
                        self.mesh.coords[2, pnt_af[idx_af_x]],
                        np.sign(ApplForce[0, pnt_af[idx_af_x]]),
                        0.,
                        0.,
                        length=ll,
                        color='b',
                        label='Applied force',
                    )
                    is_plot_af = True

                if len(idx_af_y[0]) > 0:
                    label_trc_y = None if is_plot_af else 'Applied force'
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_af[idx_af_y]],
                        self.mesh.coords[1, pnt_af[idx_af_y]],
                        self.mesh.coords[2, pnt_af[idx_af_y]],
                        0.,
                        np.sign(ApplForce[1, pnt_af[idx_af_y]]),
                        0.,
                        length=ll,
                        color='b',
                        label=label_trc_y,
                    )
                    is_plot_af = True

                if len(idx_af_z[0]) > 0:
                    label_trc_z = None if is_plot_af else 'Applied force'
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_af[idx_af_z]],
                        self.mesh.coords[1, pnt_af[idx_af_z]],
                        self.mesh.coords[2, pnt_af[idx_af_z]],
                        0.,
                        0.,
                        np.sign(ApplForce[2, pnt_af[idx_af_z]]),
                        length=ll,
                        color='b',
                        label=label_trc_z,
                    )
                    is_plot_af = True

        plt.legend()

    def set(self, *, values: dict = {}, params: dict = {}) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        values : dict
            Value to plot in each element [n_element] (1D array)
        params : dict
            'xlim' (array-like, Range of x-axis),
            'ylim' (array-like, Range of y-axis),
            'zlim' (array-like, Range of z-axis),
            'cmap' (string, Color map),
            'edgecolor' (string, Edge color),
            'lw' (int, Line width),
            'alpha' (float, Transparency)
        """

        self.set_window(params=params)

        cmap = params['cmap'] if 'cmap' in params else 'jet'
        edgecolor = params['edgecolor'] if 'edgecolor' in params else 'k'
        lw = params['lw'] if 'lw' in params else 1
        alpha = params['alpha'] if 'alpha' in params else 0.25

        verts = []
        vals = []
        value = None
        if 'val' in params:
            if params['val'] in values:
                value = values[params['val']]
        for ielm in range(self.mesh.n_element):

            for idx_nd in self.mesh.idx_face('vol', elm=ielm):
                cod = self.mesh.coords[:, np.array(self.mesh.connectivity[ielm])[idx_nd]].T
                verts.append(cod)
                if value is not None:
                    vals.append(value[ielm])

        self.pcm = Poly3DCollection(
            verts,
            facecolors='orange',
            linewidths=lw,
            edgecolors=edgecolor,
            cmap=cmap,
            alpha=alpha,
        )

        if len(vals) > 0:
            self.pcm.set_array(np.array(vals))

        self.ax.add_collection3d(self.pcm)
