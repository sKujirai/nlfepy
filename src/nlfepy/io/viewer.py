import numpy as np
from .viewer2d import Viewer2d
from .viewer3d import Viewer3d


class Viewer:
    """
    Viewer interface
    """

    def __init__(self, projection: str = None) -> None:

        if projection == '3d':
            self._viewer = Viewer3d()
        else:
            self._viewer = Viewer2d()

    def plot(self, *, mesh, val: np.ndarray = None, **kwargs) -> None:
        self._viewer.plot(mesh=mesh, val=val, **kwargs)

    def plot_bc(self, mesh, **kwargs) -> None:
        self._viewer.plot_bc(mesh=mesh, **kwargs)

    def contour(self, *, mesh, val: np.ndarray, **kwargs):
        self._viewer.contour(mesh=mesh, val=val, **kwargs)

    def show(self) -> None:
        self._viewer.show()

    def save(self, file_name, **kwargs) -> None:
        self._viewer.save(file_name, **kwargs)
