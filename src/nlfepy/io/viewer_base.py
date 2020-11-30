from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class ViewerBase(metaclass=ABCMeta):
    """
    Base class of viewer classes
    """

    def __init__(self):
        self.show_cbar = False

    @abstractmethod
    def _set_window(self, coords: np.ndarray, **kwargs) -> None:
        pass

    @abstractmethod
    def _set_bc_info(self, mesh) -> None:
        pass

    @abstractmethod
    def plot(self, *, mesh, val: np.ndarray = None, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_bc(self, mesh, **kwargs) -> None:
        pass

    @abstractmethod
    def contour(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        pass

    def show(self) -> None:
        if self.show_cbar:
            plt.colorbar(self._pcm, ax=self._ax)
        plt.show()

    def save(self, file_name) -> None:

        self._fig.savefig(file_name, transparent=True, dpi=300)
