from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


class ViewerBase(metaclass=ABCMeta):
    """
    Base class of viewer classes
    """

    def __init__(self) -> None:

        self.fig = plt.figure()

    @abstractmethod
    def set(self):
        pass

    def show(self, *, show_cbar=True):
        if show_cbar:
            plt.colorbar(self.pcm, ax=self.ax)
        plt.show()

    def save(self, file_name):

        self.fig.savefig(file_name, transparent=True, dpi=300)
