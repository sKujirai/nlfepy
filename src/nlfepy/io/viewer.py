from .viewer2d import Viewer2d
from .viewer3d import Viewer3d


class Viewer:
    """
    Viewer interface
    """

    def __init__(self, *, mesh, set_mesh_info=True) -> None:

        if mesh.n_dof == 2:
            self._viewer = Viewer2d(
                mesh=mesh,
                set_mesh_info=set_mesh_info
            )
        else:
            self._viewer = Viewer3d(
                mesh=mesh,
                set_mesh_info=set_mesh_info
            )

    def set(self, *, values: dict = {}, params: dict = {}) -> None:
        self._viewer.set(values=values, params=params)

    def show(self, *, show_cbar=True) -> None:
        self._viewer.show(show_cbar=show_cbar)

    def save(self, file_name) -> None:
        self._viewer.save(file_name)
