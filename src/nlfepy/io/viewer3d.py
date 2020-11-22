import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .viewer_base import ViewerBase


class Viewer3d(ViewerBase):

    def __init__(self) -> None:
        """
        Viewer class for 3D stuructures inheriting class: ViewerBase
        """

        super().__init__()

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

    def set(self, *, mesh, value=None, params={}) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        mesh :
            Mesh class (See mesh.py)
        value : array-like
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

        if 'xlim' in params:
            self.ax.set_xlim(params['xlim'][0], params['xlim'][1])
        else:
            self.ax.set_xlim(np.min(mesh.coords[0]), np.max(mesh.coords[0]))
        if 'ylim' in params:
            self.ax.set_ylim(params['ylim'][0], params['ylim'][1])
        else:
            self.ax.set_ylim(np.min(mesh.coords[1]), np.max(mesh.coords[1]))
        if 'zlim' in params:
            self.ax.set_zlim(params['zlim'][0], params['zlim'][1])
        else:
            self.ax.set_zlim(np.min(mesh.coords[2]), np.max(mesh.coords[2]))

        cmap = params['cmap'] if 'cmap' in params else 'jet'
        edgecolor = params['edgecolor'] if 'edgecolor' in params else 'k'
        lw = params['lw'] if 'lw' in params else 1
        alpha = params['alpha'] if 'alpha' in params else 0.25

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        verts = []
        values = []
        for ielm in range(mesh.n_element):

            for idx_nd in mesh.idx_face('vol', elm=ielm):
                cod = mesh.coords[:, np.array(mesh.connectivity[ielm])[idx_nd]].T
                verts.append(cod)
                if value is not None:
                    values.append(value[ielm])

        self.pcm = Poly3DCollection(
            verts,
            facecolors='orange',
            linewidths=lw,
            edgecolors=edgecolor,
            cmap=cmap,
            alpha=alpha,
        )

        if len(values) > 0:
            self.pcm.set_array(np.array(values))

        self.ax.add_collection3d(self.pcm)
