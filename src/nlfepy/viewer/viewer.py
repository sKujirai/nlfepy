import numpy as np
from .viewer2d import Viewer2d
from .viewer3d import Viewer3d
from ..io.vtu_reader import VtuReader
from ..mesh.mesh import Mesh


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

    def contour(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        self._viewer.contour(mesh=mesh, val=val, **kwargs)

    def multi_plot(self, file, cnfs: list, **kwargs) -> None:

        reader = VtuReader()
        reader.read(file)
        mesh = Mesh()
        mesh.set_mesh_data(
            mesh=reader.mesh,
            bc=reader.bc,
            mpc=reader.mpc
        )

        val_list = []
        for cnf in cnfs:
            sys = cnf['sys'] if 'sys' in cnf.keys() else None
            plot_mode = cnf['plot'] if 'plot' in cnf.keys() else 'fill'
            if plot_mode == 'contour':
                val = reader.get_point_value(cnf['val'], sys=sys).astype(np.float)
            else:
                val = reader.get_elm_value(cnf['val'], sys=sys).astype(np.float)
            if sys is None:
                sys = [i for i in range(val.shape[1])]
            for i, isys in enumerate(sys):
                val_list.append(
                    {
                        'val': val[:, i],
                        'figname': cnf['val'] + ' ' + str(isys),
                        'plot': plot_mode,
                    }
                )

        self._viewer.multi_plot(mesh=mesh, vlist=val_list, **kwargs)

    def show(self) -> None:
        self._viewer.show()

    def save(self, file_name, **kwargs) -> None:
        self._viewer.save(file_name, **kwargs)
