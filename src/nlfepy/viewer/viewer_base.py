from abc import ABCMeta, abstractmethod
import io
import numpy as np
import matplotlib.pyplot as plt


class ViewerBase(metaclass=ABCMeta):
    """
    Base class of viewer classes
    """

    def __init__(self):
        pass

    @abstractmethod
    def _set_window(self, ax, coords: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def _set_bc_info(self, ax, mesh):
        pass

    @abstractmethod
    def _ax_plot(self, *, ax, mesh, val: np.ndarray = None, **kwargs):
        pass

    @abstractmethod
    def _ax_contour(self, *, ax, mesh, val: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def _ax_scatter(self, *, ax, mesh, val: np.ndarray, **kwargs):
        pass

    def plot(self, *, mesh, val: np.ndarray = None, **kwargs) -> None:
        """
        Set coordinates, connectivity and values to plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        zlim : array-like
            Range of z-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        alpha : float
            Transparency
        """

        plt.close()

        title = kwargs["title"] if "title" in kwargs else None

        if "show_axis_label" not in kwargs:
            kwargs["show_axis_label"] = True

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(
            111, title=title, projection=kwargs["projection"]
        )
        self._ax, self._pcm = self._ax_plot(ax=self._ax, mesh=mesh, val=val, **kwargs)

        if val is not None:
            plt.colorbar(self._pcm, ax=self._ax)

    def plot_bc(self, mesh, **kwargs) -> None:
        """
        Plot boundary conditions

        Parameters
        ----------
        mesh :
            Mesh class
        """

        if "title" not in kwargs:
            kwargs["title"] = "Boundary conditions"

        if "show_axis_label" not in kwargs:
            kwargs["show_axis_label"] = True

        self.plot(mesh=mesh, **kwargs)

        self._ax = self._set_bc_info(self._ax, mesh)

    def contour(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        """
        Contour plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        zlim : array-like
            Range of z-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        alpha : float
            Transparency
        """

        plt.close()

        title = kwargs["title"] if "title" in kwargs else None

        if "show_axis_label" not in kwargs:
            kwargs["show_axis_label"] = True

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(
            111, title=title, projection=kwargs["projection"]
        )
        self._ax, self._pcm = self._ax_contour(
            ax=self._ax, mesh=mesh, val=val, **kwargs
        )

        if val is not None:
            plt.colorbar(self._pcm, ax=self._ax)

    def scatter(self, *, mesh, val: np.ndarray, **kwargs) -> None:
        """
        Scatter plot

        Parameters
        ----------
        mesh :
            Mesh class
        val : ndarray
            Value to plot in each element [n_element] (1D array)
        xlim : array-like
            Range of x-axis
        ylim : array-like
            Range of y-axis
        zlim : array-like
            Range of z-axis
        cmap : str
            Color map
        edgecolor : str
            Edge color
        lw : int
            Line width
        alpha : float
            Transparency
        """

        plt.close()

        title = kwargs["title"] if "title" in kwargs else None

        if "show_axis_label" not in kwargs:
            kwargs["show_axis_label"] = True

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(
            111, title=title, projection=kwargs["projection"]
        )
        self._delete_plt_unnecessary_keys(kwargs)
        self._ax, self._pcm = self._ax_scatter(
            ax=self._ax, mesh=mesh, val=val, **kwargs
        )

        if val is not None:
            plt.colorbar(self._pcm, ax=self._ax)

    def _delete_plt_unnecessary_keys(self, fkeys: dict):
        if "title" in fkeys:
            del fkeys["title"]
        if "show_axis_label" in fkeys:
            del fkeys["show_axis_label"]
        if "projection" in fkeys:
            del fkeys["projection"]
        if "overlay" in fkeys:
            del fkeys["overlay"]

    def multi_plot(self, mesh, vlist, **kwargs) -> None:
        """
        Plot multiple figures

        Parameters
        ----------
        mesh :
            Mesh class
        vlist: list
            List of configurations.
            'val': Value to plot
            'plot': Plot mode ('fill', 'contour' or 'scatter')
            'figname': Figure name
        """

        if "overlay" not in kwargs or self._fig is None:
            kwargs["overlay"] = False

        n_fig = len(vlist)
        n_col = kwargs["max_ncol"] if "max_ncol" in kwargs.keys() else min(n_fig, 3)
        n_row = -(-n_fig // n_col)

        if not kwargs["overlay"]:
            plt.close()
            self._fig = plt.figure(figsize=(3 * n_col, 3 * n_row))

            self._ax = []
            self._pcm = []

        for i, vl in enumerate(vlist):
            if not kwargs["overlay"]:
                ax = self._fig.add_subplot(
                    n_row, n_col, i + 1, projection=kwargs["projection"]
                )
            else:
                ax = self._ax[i]
            ax.set_title(vl["figname"])
            if vl["plot"] == "contour":
                ax, pcm = self._ax_contour(ax=ax, mesh=mesh, val=vl["val"], **kwargs)
            elif vl["plot"] == "scatter":
                ax, pcm = self._ax_scatter(ax=ax, mesh=mesh, val=vl["val"], **kwargs)
            else:
                ax, pcm = self._ax_plot(ax=ax, mesh=mesh, val=vl["val"], **kwargs)
            if not kwargs["overlay"]:
                self._ax.append(ax)
                self._pcm.append(pcm)
                self._fig.colorbar(pcm, ax=ax)

    def show(self) -> None:

        plt.show()

    def save(self, file_name, **kwargs) -> None:

        self._fig.savefig(file_name, **kwargs)

    def get_fig_array(self, **kwargs) -> np.ndarray:
        """
        Get figure as ndarray

        Returns
        -------
        img_arr : np.ndarray
            Numpy array of current figure
        """

        dpi = kwargs["dpi"] if "dpi" in kwargs else 300
        io_buf = io.BytesIO()
        self._fig.savefig(io_buf, format="raw", dpi=dpi)
        io_buf.seek(0)
        img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        bbox = self._fig.get_window_extent().transformed(
            self._fig.dpi_scale_trans.inverted()
        )
        width, height = int(bbox.width), int(bbox.height)
        l_unit = int(np.sqrt(img_arr.shape[0] / (4 * height * width)))
        img_arr = img_arr.reshape(l_unit * height, l_unit * width, 4)
        io_buf.close()

        return img_arr
