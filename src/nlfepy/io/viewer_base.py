from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


class ViewerBase(metaclass=ABCMeta):
    """
    Base class of viewer classes
    """

    def __init__(self, *, mesh) -> None:
        """
        Initialization

        Parameters
        ----------
        mesh :
            Mesh class (See mesh.py)
        """

        self._mesh = mesh

    @abstractmethod
    def _set_window(self, *, params: dict = {}) -> None:
        pass

    @abstractmethod
    def _set_bc_info(self) -> None:
        pass

    @abstractmethod
    def set(self, *, values: dict = {}, params: dict = {}) -> None:
        pass

    def show(self, *, show_cbar=True) -> None:
        if show_cbar:
            plt.colorbar(self._pcm, ax=self._ax)
        plt.show()

    def save(self, file_name) -> None:

        self._fig.savefig(file_name, transparent=True, dpi=300)
