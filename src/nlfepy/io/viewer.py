from .viewer2d import Viewer2d
from .viewer3d import Viewer3d


class Viewer:
    """
    Viewer interface
    """

    def __init__(self, *, mesh, params: dict = {}) -> None:
        self.mesh = mesh
        self.params = params

        if self.mesh.n_dof == 2:
            self.viewer = Viewer2d()
        else:
            self.viewer = Viewer3d()

    def set(self, *, value=None) -> None:
        self.viewer.set(mesh=self.mesh, value=value, params=self.params)

    def show(self, *, show_cbar=True):
        self.viewer.show(show_cbar=show_cbar)

    def save(self, file_name):
        self.viewer.save(file_name)
