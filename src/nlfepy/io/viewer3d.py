import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .viewer import Viewer


class Viewer3d(Viewer):

    def __init__(self) -> None:
        """
        Viewer class for 3D stuructures inheriting class: Viewer
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

    def set(self, coords: np.ndarray, connectivity: np.ndarray, shapef, value=None, xlim=None, ylim=None, zlim=None, cmap='jet', edgecolor='k', lw=1, alpha=0.25) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        coords : ndarray
            Coordinates [3, npoint] (2D array)
        connectivity : array-like
            Connectivity of elaments [n_element][n_node] (2D array)
        shapef :
            Shape function class
        value : array-like
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        zlim : array-like
            Range of z-axis
        cmap : string
            Color map
        edgecolor : string
            Edge color
        lw : int
            Line width
        alpha : float
            Transparency
        """

        if xlim is None:
            self.ax.set_xlim(np.min(coords[0]), np.max(coords[0]))
        else:
            self.ax.set_xlim(xlim[0], xlim[1])
        if ylim is None:
            self.ax.set_ylim(np.min(coords[1]), np.max(coords[1]))
        else:
            self.ax.set_ylim(ylim[0], ylim[1])
        if zlim is None:
            self.ax.set_zlim(np.min(coords[2]), np.max(coords[2]))
        else:
            self.ax.set_zlim(zlim[0], zlim[1])

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        verts = []
        values = []
        for ielm in range(connectivity.shape[0]):
            for ifce in range(shapef['vol'].n_face):
                cod = coords[:, connectivity[ielm][shapef['vol'].idx_face[ifce]]][:3].T
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
