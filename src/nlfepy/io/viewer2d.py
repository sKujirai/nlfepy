import sys
from logging import getLogger
import numpy as np
from matplotlib.collections import PolyCollection
from .viewer import Viewer


class Viewer2d(Viewer):
    """
    Viewer class for 2D stuructures inheriting class: Viewer
    """

    def __init__(self) -> None:

        super().__init__()

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
        self.ax.set_aspect('equal')

    def set(self, coords: np.ndarray, connectivity: np.ndarray, value=None, xlim=None, ylim=None, cmap='jet', edgecolor='k', lw=1) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        coords : ndarray
            Coordinates [2, npoint] (2D array)
        connectivity : array-like
            Connectivity of elaments [n_element][n_node] (2D array)
        value : array-like
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        cmap : string
            Color map
        edgecolor : string
            Edge color
        lw : int
            Line width
        """

        if xlim is None:
            self.ax.set_xlim(np.min(coords[0]), np.max(coords[0]))
        else:
            self.ax.set_xlim(xlim[0], xlim[1])
        if ylim is None:
            self.ax.set_ylim(np.min(coords[1]), np.max(coords[1]))
        else:
            self.ax.set_ylim(ylim[0], ylim[1])

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        vertices = coords[:2, :].T[np.asarray(connectivity)]

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
            if value.shape[0] != connectivity.shape[0]:
                logger = getLogger('viewer2d')
                logger.error('Value arrray has invalid size. Size of values: {}, Number of elements: {}'.format(value.shape[0], connectivity.shape[0]))
                sys.exit(1)
            self.pcm.set_array(value)
        self.ax.add_collection(self.pcm)
