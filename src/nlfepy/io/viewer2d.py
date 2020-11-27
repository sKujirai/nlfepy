import sys
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from .viewer_base import ViewerBase


class Viewer2d(ViewerBase):
    """
    Viewer class for 2D stuructures inheriting class: ViewerBase
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
            'cmap' (string, Color map),
            'edgecolor' (string, Edge color),
            'lw' (int, Line width)
        """

        plt.close()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # self.ax.axis('off')
        self.ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )
        self.ax.axes.xaxis.set_ticks([])
        self.ax.axes.yaxis.set_ticks([])
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.set_aspect('equal')

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        ratio_margin = 0.1
        specimen_size = np.max(self.mesh.coords[:self.mesh.n_dof], axis=1)

        if 'xlim' in params:
            self.ax.set_xlim(params['xlim'][0], params['xlim'][1])
        else:
            self.ax.set_xlim(np.min(self.mesh.coords[0]) - specimen_size[0]*ratio_margin, np.max(self.mesh.coords[0]) + specimen_size[0]*ratio_margin)
        if 'ylim' in params:
            self.ax.set_ylim(params['ylim'][0], params['ylim'][1])
        else:
            self.ax.set_ylim(np.min(self.mesh.coords[1]) - specimen_size[1]*ratio_margin, np.max(self.mesh.coords[1]) + specimen_size[1]*ratio_margin)

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
        fix_pts, fix_dof = np.divmod(self.mesh.bc['idx_fix'], self.mesh.n_dof)
        fix_x = fix_pts[fix_dof == 0]
        fix_y = fix_pts[fix_dof == 1]
        fix_x = np.setdiff1d(fix_x, disp_pts[idx_disp_x])
        fix_y = np.setdiff1d(fix_y, disp_pts[idx_disp_y])
        fix_xy = np.intersect1d(fix_x, fix_y)
        fix_x = np.setdiff1d(fix_x, fix_xy)
        fix_y = np.setdiff1d(fix_y, fix_xy)

        ll = max(np.max(self.mesh.coords[:self.mesh.n_dof], axis=1)) * rr

        # Fix points
        self.ax.scatter(
            self.mesh.coords[0, fix_xy],
            self.mesh.coords[1, fix_xy],
            color='r',
            label='Completely fixed',
        )

        if len(fix_x) > 0:
            self.ax.scatter(
                self.mesh.coords[0, fix_x],
                self.mesh.coords[1, fix_x],
                color='b',
                label='Fixed in x direction',
            )

        if len(fix_y) > 0:
            self.ax.scatter(
                self.mesh.coords[0, fix_y],
                self.mesh.coords[1, fix_y],
                color='g',
                label='Fixed in y direction',
            )

        # Prescribed displacement
        ratio_disp = ll / abs(np.max(self.mesh.bc['displacement'])) * rr if len(self.mesh.bc['idx_disp']) > 0 else None

        if len(idx_disp_x[0]) > 0:
            self.ax.quiver(
                self.mesh.coords[0, disp_pts[idx_disp_x]],
                self.mesh.coords[1, disp_pts[idx_disp_x]],
                self.mesh.bc['displacement'][idx_disp_x]*ratio_disp,
                0.,
                color='r',
                label='Prescribed displacement',
            )

        if len(idx_disp_y[0]) > 0:
            label_disp_y = None if len(idx_disp_x[0]) > 0 else 'Prescribed displacement'
            self.ax.quiver(
                self.mesh.coords[0, disp_pts[idx_disp_y]],
                self.mesh.coords[1, disp_pts[idx_disp_y]],
                0.,
                self.mesh.bc['displacement'][idx_disp_y]*ratio_disp,
                color='r',
                label=label_disp_y,
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
                ratio_trc = ll / max_trc * rr

                if len(idx_trc_x[0]) > 0:
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_trc[idx_trc_x]],
                        self.mesh.coords[1, pnt_trc[idx_trc_x]],
                        Traction[0, pnt_trc[idx_trc_x]] * ratio_trc,
                        0.,
                        color='b',
                        label='Traction',
                    )

                if len(idx_trc_y[0]) > 0:
                    label_trc_y = None if len(idx_trc_x[0]) > 0 else 'Traction'
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_trc[idx_trc_y]],
                        self.mesh.coords[1, pnt_trc[idx_trc_y]],
                        0.,
                        Traction[1, pnt_trc[idx_trc_y]] * ratio_trc,
                        color='b',
                        label=label_trc_y,
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
                ratio_af = ll / max_af * rr

                if len(idx_af_x[0]) > 0:
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_af[idx_af_x]],
                        self.mesh.coords[1, pnt_af[idx_af_x]],
                        ApplForce[0, pnt_af[idx_af_x]]*ratio_af,
                        0.,
                        color='b',
                        label='Applied force',
                    )

                if len(idx_af_y[0]) > 0:
                    label_trc_y = None if len(idx_af_y[0]) > 0 else 'Applied force'
                    self.ax.quiver(
                        self.mesh.coords[0, pnt_af[idx_af_y]],
                        self.mesh.coords[1, pnt_af[idx_af_y]],
                        0.,
                        ApplForce[1, pnt_af[idx_af_y]]*ratio_af,
                        color='b',
                        label=label_trc_y,
                    )

        self.ax.legend()

    def set(self, *, values: dict = {}, params: dict = {}) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        value : dict
            Value to plot in each element [n_element] (1D array)
        params : dict
            'xlim' (array-like, Range of x-axis),
            'ylim' (array-like, Range of y-axis),
            'cmap' (string, Color map),
            'edgecolor' (string, Edge color),
            'lw' (int, Line width)
        """

        self.set_window(params=params)

        cmap = params['cmap'] if 'cmap' in params else 'jet'
        edgecolor = params['edgecolor'] if 'edgecolor' in params else 'k'
        lw = params['lw'] if 'lw' in params else 1

        vertices = self.mesh.coords[:2, :].T[np.asarray(self.mesh.connectivity)]

        value = None
        if 'val' in params:
            if params['val'] in values:
                value = values[params['val']]

        if value is None:
            self.pcm = PolyCollection(
                vertices,
                facecolor='None',
                edgecolors=edgecolor,
                linewidths=lw,
                cmap=cmap,
            )
        else:
            self.pcm = PolyCollection(
                vertices,
                edgecolors=edgecolor,
                linewidths=lw,
                cmap=cmap,
            )
            value = np.array(value)
            if value.shape[0] != self.mesh.n_element:
                logger = getLogger('viewer2d')
                logger.error('Value arrray has invalid size. Size of values: {}, Number of elements: {}'.format(value.shape[0], self.mesh.n_element))
                sys.exit(1)
            self.pcm.set_array(value)
        self.ax.add_collection(self.pcm)
