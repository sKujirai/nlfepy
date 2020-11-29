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
            self._set_bc_info()

    def _set_window(self, *, params: dict = {}) -> None:
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
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        # self._ax.axis('off')
        self._ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=False,
            left=False,
            right=False,
            top=False,
        )
        self._ax.axes.xaxis.set_ticks([])
        self._ax.axes.yaxis.set_ticks([])
        self._ax.spines['left'].set_visible(False)
        self._ax.spines['bottom'].set_visible(False)
        self._ax.spines['right'].set_visible(False)
        self._ax.spines['top'].set_visible(False)
        self._ax.set_aspect('equal')

        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        ratio_margin = 0.1
        specimen_size = np.max(self._mesh.coords[:self._mesh.n_dof], axis=1)

        if 'xlim' in params:
            self._ax.set_xlim(params['xlim'][0], params['xlim'][1])
        else:
            self._ax.set_xlim(np.min(self._mesh.coords[0]) - specimen_size[0]*ratio_margin, np.max(self._mesh.coords[0]) + specimen_size[0]*ratio_margin)
        if 'ylim' in params:
            self._ax.set_ylim(params['ylim'][0], params['ylim'][1])
        else:
            self._ax.set_ylim(np.min(self._mesh.coords[1]) - specimen_size[1]*ratio_margin, np.max(self._mesh.coords[1]) + specimen_size[1]*ratio_margin)

    def _set_bc_info(self) -> None:
        """
        Plot boundary conditions
        """

        rr = 0.2
        eps_crit = 1.e-10
        eps_min = 1.e-30

        disp_pts, disp_dof = np.divmod(self._mesh.bc['idx_disp'], self._mesh.n_dof)
        idx_disp_x = np.where(disp_dof == 0)
        idx_disp_y = np.where(disp_dof == 1)
        fix_pts, fix_dof = np.divmod(self._mesh.bc['idx_fix'], self._mesh.n_dof)
        fix_x = fix_pts[fix_dof == 0]
        fix_y = fix_pts[fix_dof == 1]
        fix_x = np.setdiff1d(fix_x, disp_pts[idx_disp_x])
        fix_y = np.setdiff1d(fix_y, disp_pts[idx_disp_y])
        fix_xy = np.intersect1d(fix_x, fix_y)
        fix_x = np.setdiff1d(fix_x, fix_xy)
        fix_y = np.setdiff1d(fix_y, fix_xy)

        ll = max(np.max(self._mesh.coords[:self._mesh.n_dof], axis=1)) * rr

        # Fix points
        self._ax.scatter(
            self._mesh.coords[0, fix_xy],
            self._mesh.coords[1, fix_xy],
            color='r',
            label='Completely fixed',
        )

        if len(fix_x) > 0:
            self._ax.scatter(
                self._mesh.coords[0, fix_x],
                self._mesh.coords[1, fix_x],
                color='b',
                label='Fixed in x direction',
            )

        if len(fix_y) > 0:
            self._ax.scatter(
                self._mesh.coords[0, fix_y],
                self._mesh.coords[1, fix_y],
                color='g',
                label='Fixed in y direction',
            )

        # Prescribed displacement
        ratio_disp = ll / abs(np.max(self._mesh.bc['displacement'])) * rr if len(self._mesh.bc['idx_disp']) > 0 else None

        if len(idx_disp_x[0]) > 0:
            self._ax.quiver(
                self._mesh.coords[0, disp_pts[idx_disp_x]],
                self._mesh.coords[1, disp_pts[idx_disp_x]],
                self._mesh.bc['displacement'][idx_disp_x]*ratio_disp,
                0.,
                color='r',
                label='Prescribed displacement',
            )

        if len(idx_disp_y[0]) > 0:
            label_disp_y = None if len(idx_disp_x[0]) > 0 else 'Prescribed displacement'
            self._ax.quiver(
                self._mesh.coords[0, disp_pts[idx_disp_y]],
                self._mesh.coords[1, disp_pts[idx_disp_y]],
                0.,
                self._mesh.bc['displacement'][idx_disp_y]*ratio_disp,
                color='r',
                label=label_disp_y,
            )

        # Traction
        if 'traction' in self._mesh.bc:
            Traction = self._mesh.bc['traction']
            max_trc = np.max(np.abs(Traction))
            if max_trc > eps_min:
                trc_crit = max_trc * eps_crit
                pnt_trc, dof_trc = np.where(np.abs(Traction) > trc_crit)
                idx_trc_x = np.where(dof_trc == 0)
                idx_trc_y = np.where(dof_trc == 1)
                ratio_trc = ll / max_trc * rr

                if len(idx_trc_x[0]) > 0:
                    self._ax.quiver(
                        self._mesh.coords[0, pnt_trc[idx_trc_x]],
                        self._mesh.coords[1, pnt_trc[idx_trc_x]],
                        Traction[pnt_trc[idx_trc_x], 0] * ratio_trc,
                        0.,
                        color='g',
                        label='Traction',
                    )

                if len(idx_trc_y[0]) > 0:
                    label_trc_y = None if len(idx_trc_x[0]) > 0 else 'Traction'
                    self._ax.quiver(
                        self._mesh.coords[0, pnt_trc[idx_trc_y]],
                        self._mesh.coords[1, pnt_trc[idx_trc_y]],
                        0.,
                        Traction[pnt_trc[idx_trc_y], 1] * ratio_trc,
                        color='g',
                        label=label_trc_y,
                    )

        # Applied force
        if 'applied_force' in self._mesh.bc:
            ApplForce = self._mesh.bc['applied_force']
            max_af = np.max(np.abs(ApplForce))
            if max_af > eps_min:
                af_crit = max_af * eps_crit
                pnt_af, dof_af = np.where(np.abs(ApplForce) > af_crit)
                idx_af_x = np.where(dof_af == 0)
                idx_af_y = np.where(dof_af == 1)
                ratio_af = ll / max_af * rr

                if len(idx_af_x[0]) > 0:
                    self._ax.quiver(
                        self._mesh.coords[0, pnt_af[idx_af_x]],
                        self._mesh.coords[1, pnt_af[idx_af_x]],
                        ApplForce[pnt_af[idx_af_x], 0]*ratio_af,
                        0.,
                        color='b',
                        label='Applied force',
                    )

                if len(idx_af_y[0]) > 0:
                    label_trc_y = None if len(idx_af_y[0]) > 0 else 'Applied force'
                    self._ax.quiver(
                        self._mesh.coords[0, pnt_af[idx_af_y]],
                        self._mesh.coords[1, pnt_af[idx_af_y]],
                        0.,
                        ApplForce[pnt_af[idx_af_y], 1]*ratio_af,
                        color='b',
                        label=label_trc_y,
                    )

        self._ax.legend()

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

        self._set_window(params=params)

        cmap = params['cmap'] if 'cmap' in params else 'jet'
        edgecolor = params['edgecolor'] if 'edgecolor' in params else 'k'
        lw = params['lw'] if 'lw' in params else 1

        vertices = self._mesh.coords[:2, :].T[np.asarray(self._mesh.connectivity)]

        value = None
        if 'val' in params:
            if params['val'] in values:
                value = values[params['val']]

        if value is None:
            self._pcm = PolyCollection(
                vertices,
                facecolor='None',
                edgecolors=edgecolor,
                linewidths=lw,
                cmap=cmap,
            )
        else:
            self._pcm = PolyCollection(
                vertices,
                edgecolors=edgecolor,
                linewidths=lw,
                cmap=cmap,
            )
            value = np.array(value)
            if value.shape[0] != self._mesh.n_element:
                logger = getLogger('viewer2d')
                logger.error('Value arrray has invalid size. Size of values: {}, Number of elements: {}'.format(value.shape[0], self._mesh.n_element))
                sys.exit(1)
            self._pcm.set_array(value)
        self._ax.add_collection(self._pcm)
