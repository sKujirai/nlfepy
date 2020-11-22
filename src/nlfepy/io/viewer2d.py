import sys
from logging import getLogger
import numpy as np
from matplotlib.collections import PolyCollection
from .viewer_base import ViewerBase


class Viewer2d(ViewerBase):
    """
    Viewer class for 2D stuructures inheriting class: ViewerBase
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
            'cmap' (string, Color map),
            'edgecolor' (string, Edge color),
            'lw' (int, Line width)
        """

        if 'xlim' in params:
            self.ax.set_xlim(params['xlim'][0], params['xlim'][1])
        else:
            self.ax.set_xlim(np.min(mesh.coords[0]), np.max(mesh.coords[0]))
        if 'ylim' in params:
            self.ax.set_ylim(params['ylim'][0], params['ylim'][1])
        else:
            self.ax.set_ylim(np.min(mesh.coords[1]), np.max(mesh.coords[1]))

        cmap = params['cmap'] if 'cmap' in params else 'jet'
        edgecolor = params['edgecolor'] if 'edgecolor' in params else 'k'
        lw = params['lw'] if 'lw' in params else 1

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        vertices = mesh.coords[:2, :].T[np.asarray(mesh.connectivity)]

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
            if value.shape[0] != mesh.n_element:
                logger = getLogger('viewer2d')
                logger.error('Value arrray has invalid size. Size of values: {}, Number of elements: {}'.format(value.shape[0], mesh.n_element))
                sys.exit(1)
            self.pcm.set_array(value)
        self.ax.add_collection(self.pcm)
