import numpy as np
from typing import Tuple
from abc import ABCMeta, abstractmethod


class ConstitutiveBase(metaclass=ABCMeta):
    """
    Constitutive class (base class)

    Attributes
    ----------
    metal :
        Metal class inheriting material class (See metal.py)
    ntintgp : int
        Number of total integral points
    val : dict
        Dictionary of physical quantities on integral points
    params : dict
        Parameters in the constitutive equation
    """

    def __init__(self, *, metal, nitg: int, val: dict = {}, params: dict = {}) -> None:

        self._metal = metal
        self._ntintgp = nitg
        self._val = val
        self._params = params

        if "cmatrix" not in self._val:
            self._val["cmatrix"] = np.tile(self._metal.Cmatrix, (self._ntintgp, 1, 1))
        if "rvector" not in self._val:
            self._val["rvector"] = np.zeros((self._ntintgp, 6))
        if "stress" not in self._val:
            self._val["stress"] = np.zeros((self._ntintgp, 6))
        if "thickness" not in self._val:
            self._val["thickness"] = np.ones(self._ntintgp)

        if "dt" not in self._params:
            self._params["dt"] = 0.01

    @abstractmethod
    def constitutive_equation(
        self, *, du: np.ndarray, bm: np.ndarray, itg: int, plane_stress_type: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve constitutive equation

        Parameters
        ----------
        du : ndarray
            Incremental displacement [n_dof, n_node]
        bm : ndarray
            B-matrix
        itg : int
            Index of integral point
        plane_stress_type : int
            Type of plane stress for 2D
            0: Plane strain
            1: Plane stress (T33 = T23 = T31 = 0)
            2: Plane stress (E23 = E31 = T33 = 0)

        Returns
        -------
        Cmatrix : ndarray
            Elastic modulus matrix, Cij (6 x 6)
        Rvector : ndarray
            R-vector, Ri (6)
        Stress : ndarray
            Stress vector, Ti (6)
        """
        pass

    def get_ctensor(self, *, Cij: np.ndarray) -> np.ndarray:
        """
        Cij [6, 6] -> Cijkl [3, 3, 3, 3]

        Parameters
        ----------
        Cij : ndarray
            Elastic modulus matrix Cij (6 x 6)

        Returns
        -------
        Cijkl : ndarray
            Elastic modulus tensor Cijkl (3 x 3 x 3 x 3)
        """

        C99 = np.concatenate([Cij, Cij[:, 3:6]], axis=1)
        C99 = np.concatenate([C99, C99[3:6, :]], axis=0)
        C99[[0, 3, 8, 6, 1, 4, 5, 7, 2], :][:, [0, 3, 8, 6, 1, 4, 5, 7, 2]]
        Cijkl = C99[[0, 3, 8, 6, 1, 4, 5, 7, 2], :][
            :, [0, 3, 8, 6, 1, 4, 5, 7, 2]
        ].reshape(3, 3, 3, 3)

        return Cijkl

    def get_cmatrix(self, *, Cijkl: np.ndarray) -> np.ndarray:
        """
        Cijkl [3, 3, 3, 3] -> Cij [6, 6]

        Parameters
        ----------
        Cijkl : ndarray
            Elastic modulus tensor Cijkl (3 x 3 x 3 x 3)

        Returns
        -------
        Cij : ndarray
            Elastic modulus matrix Cij (6 x 6)
        """

        return Cijkl.reshape(9, 9)[[0, 4, 8, 1, 5, 6, 3, 7, 2], :][
            :, [0, 4, 8, 1, 5, 6, 3, 7, 2]
        ][:6, :6]

    def set_thickness(self, thickness: float) -> None:
        """
        Set thickness of specimen for plane stress condition

        Parameters
        ----------
        thickness : float
            Thickness of specimen
        """

        self._val["thickness"] = np.full(self._ntintgp, thickness)

    def get_thickness(self, itg: int) -> float:
        """
        Get thickness of specimen for plane stress condition

        Parameters
        ----------
        itg : int
            Index of integral point

        Returns
        -------
        thickness : float
            Thickness of specimen
        """

        return self._val["thickness"][itg]

    def get_elastic_modulus(self, n_dof: int, plane_stress_type: int = 0) -> np.ndarray:
        """
        Get elastic modulus matrix

        Parameters
        ----------
        n_dof : int
            Number of degrees of freedom
        plane_stress_type : int
            Type of plane stress for 2D
            0: Plane strain
            1: Plane stress (T33 = T23 = T31 = 0)
            2: Plane stress (E23 = E31 = T33 = 0)

        Returns
        -------
        Ce : ndarray
            Elastic modulus matrix
        """

        if n_dof == 3:
            return self._metal.Cmatrix
        else:
            C124 = self._metal.Cmatrix[[0, 1, 3], :][:, [0, 1, 3]]
            if plane_stress_type == 1:
                return (
                    C124
                    - np.tensordot(
                        self._metal.Cmatrix[[0, 1, 3], 2],
                        self._metal.Cmatrix[2, [0, 1, 3]],
                        axes=0,
                    )
                    / self._metal.Cmatrix[2, 2]
                )
            elif plane_stress_type == 2:
                C356 = self._metal.Cmatrix[[2, 4, 5], :][:, [2, 4, 5]]
                C356_124 = self._metal.Cmatrix[[2, 4, 5], :][:, [0, 1, 3]]
                C124_356 = self._metal.Cmatrix[[0, 1, 3], :][:, [2, 4, 5]]
                return C124 - np.dot(np.dot(np.linalg.inv(C356), C356_124), C124_356)
            else:
                return C124

    def calc_stress_elastic(
        self,
        *,
        u_disp: np.ndarray,
        bm: np.ndarray,
        itg: int,
        plane_stress_type: int = 0
    ) -> None:
        """
        Calc. stress in the elastic region

        Parameters
        ----------
        u_disp : ndarray
            Displacement of each element [n_dof, n_node]
        bm : ndarray
            B-matrix
        itg : int
            Index of integral point
        """

        n_dof = u_disp.shape[0]

        Ee = np.dot(bm, u_disp.T.flatten())
        Ce = self.get_elastic_modulus(n_dof, plane_stress_type)
        Stress = np.dot(Ce, Ee)

        self._val["stress"][itg] = 0.0
        self._val["stress"][itg, :2] = Stress[:2]
        if n_dof == 2:
            self._val["stress"][itg, 3] = Stress[2]
        else:
            self._val["stress"][itg, 2:] = Stress[2:]

    def calc_correction_term_plane_stress_CR(
        self, Cmatrix: np.ndarray, Rvector: np.ndarray, plane_stress_type: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calc. correction term for plane stress condition

        Parameters
        ----------
        Cmatrix : ndarray
            Elastic modulus matrix, Cij (6 x 6)
        Rvector : ndarray
            R-vector, Ri (6)
        plane_stress_type : int
            Type of plane stress for 2D
            0: Plane strain
            1: Plane stress (T33 = T23 = T31 = 0)
            2: Plane stress (E23 = E31 = T33 = 0)

        Returns
        -------
        C_corr : ndarray
            Corrected elastic modulus matrix, Cij (3 x 3)
        R_corr : ndarray
            Corrected R-vector, Ri (3)
        """

        C_corr = Cmatrix[[0, 1, 3], :][:, [0, 1, 3]]
        R_corr = Rvector[[0, 1, 3]]

        if plane_stress_type == 1:
            Cij33 = Cmatrix[[0, 1, 3], 2]
            C33kl = Cmatrix[2, [0, 1, 3]]
            C3333 = Cmatrix[2, 2]
            C_corr -= np.tensordot(Cij33, C33kl, axes=0) / C3333
            R_corr -= Cij33 / C3333 * Rvector[2]
        elif plane_stress_type == 2:
            C356 = self._metal.Cmatrix[[2, 4, 5], :][:, [2, 4, 5]]
            C356_124 = self._metal.Cmatrix[[2, 4, 5], :][:, [0, 1, 3]]
            C124_356 = self._metal.Cmatrix[[0, 1, 3], :][:, [2, 4, 5]]
            Ctmp = np.dot(np.linalg.inv(C356), C356_124)
            C_corr -= np.dot(Ctmp, C124_356)
            R_corr -= np.dot(Ctmp, Rvector[[2, 4, 5]])

        return C_corr, R_corr

    def calc_correction_term_plane_stress_L(
        self, Ltensor: np.ndarray, itg: int, plane_stress_type: int = 0
    ) -> np.ndarray:
        """
        Calc. correction term for plane stress condition

        Parameters
        ----------
        Ltensor : ndarray
            Velocity gradient tensor, Lij (3 x 3)
        itg : int
            Index of integral point
        plane_stress_type : int
            Type of plane stress for 2D
            0: Plane strain
            1: Plane stress (T33 = T23 = T31 = 0)
            2: Plane stress (E23 = E31 = T33 = 0)

        Returns
        -------
        L_corr : ndarray
            Corrected velocity gradient tensor, Lij (3 x 3)
        """

        L_corr = Ltensor
        EPS = 1.0e-15

        if plane_stress_type == 1:
            if self._val["cmatrix"][itg, 2, 2] > EPS:
                L_corr[2, 2] = (
                    -1.0
                    / self._val["cmatrix"][itg, 2, 2]
                    * (
                        self._val["cmatrix"][itg, 2, 0] * L_corr[0, 0]
                        + self._val["cmatrix"][itg, 2, 1] * L_corr[1, 1]
                        + self._val["cmatrix"][itg, 2, 3] * L_corr[0, 1]
                        - self._val["rvector"][itg, 2]
                        + self._val["stress"][itg, 2]
                    )
                )
        elif plane_stress_type == 2:
            C356 = self._val["cmatrix"][itg, [2, 4, 5], :][:, [2, 4, 5]]
            if np.linalg.det(C356) > EPS:
                C356inv = np.linalg.inv(C356)
                C356_124 = self._val["cmatrix"][itg, [2, 4, 5], :][:, [0, 1, 3]]
                D3 = np.array(Ltensor[0, 0], Ltensor[1, 1], 2.0 * Ltensor[0, 1])
                R356 = self._val["rvector"][itg, [2, 4, 5]] - np.dot(C356_124, D3)
                W12 = 0.5 * (Ltensor[0, 1] - Ltensor[1, 0])
                T356 = self._val["stress"][itg, [2, 4, 5]]
                R356[0] -= T356[0]
                R356[1] -= T356[1] - W12 * T356[2]
                R356[2] -= T356[2] + W12 * T356[1]
                D356 = np.dot(C356inv, R356)
                L_corr[2, 2] = D356[0]
                L_corr[1, 2] = L_corr[2, 1] = 0.5 * D356[1]
                L_corr[2, 0] = L_corr[0, 2] = 0.5 * D356[2]

        return L_corr
