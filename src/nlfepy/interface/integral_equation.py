import numpy as np
import logging
from logging import getLogger
from abc import ABCMeta, abstractmethod


class IntegralEquation(metaclass=ABCMeta):
    """
    Base class to solve integral equation


    Attributes
    ----------
    mesh :
        Mesh class (See mesh.py)
    cnst :
        Constitutive equation class (See constitutive_base.py)
    val : dict
        Physical quantities on each node
    params : dict
        Parameters to solve governing equation
        penalty_coefficient : Penalty coefficient to solve KU=F using penalty method
        plane_stress : 0: Plane strain, 1: Plane stress (T33 = T23 = T31 = 0), 2: Plane stress (E23 = E31 = T33 = 0)
    """

    def __init__(self, *, mesh, cnst, val=None, params: dict = {}) -> None:

        self._mesh = mesh
        self._cnst = cnst
        self._val = val

        # Set parameters
        self._config = {}
        self._config["penalty_coefficient"] = 1.0e8
        self._config["logging"] = False
        self._config["plane_stress"] = 0
        self._config["thickness"] = 1.0

        for key, value in params.items():
            self._config[key] = value

        # Plane stress condition
        if self._config["plane_stress"] > 0 and "thickness" in self._config.keys():
            self._cnst[0].set_thickness(self._config["thickness"])

        # Set variables
        if "u_disp" not in self._val:
            self._val["u_disp"] = np.zeros((self._mesh.n_dof, self._mesh.n_point))
        if "deltau" not in self._val:
            self._val["deltau"] = np.zeros((self._mesh.n_dof, self._mesh.n_point))

        # Set logger
        self._logger = getLogger("ItgEqn")
        ch = logging.StreamHandler()
        if self._config["logging"]:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
        self._logger.addHandler(ch)

    @abstractmethod
    def solve(self) -> None:
        pass

    def calc_stress(self) -> None:
        """
        Calc. stress in the elastic region
        """

        n_dof = self._mesh.n_dof
        n_dfdof = self._mesh.n_dfdof
        n_element = self._mesh.n_element
        connectivity = self._mesh.connectivity

        for ielm in range(n_element):

            mater_id = self._mesh.material_numbers[ielm]
            n_node_v = self._mesh.n_node("vol", elm=ielm)
            n_intgp_v = self._mesh.n_intgp("vol", elm=ielm)
            u_elm = self._val["u_disp"][:, np.array(connectivity[ielm])]

            for itg in range(n_intgp_v):

                Bmatrix, _ = self._mesh.get_Bmatrix("vol", elm=ielm, itg=itg)
                Bd = np.zeros((n_dfdof, n_dof * n_node_v))
                if n_dof == 2:
                    Bd[0, ::n_dof] = Bmatrix[0]
                    Bd[2, ::n_dof] = Bmatrix[1]
                    Bd[1, 1::n_dof] = Bmatrix[1]
                    Bd[2, 1::n_dof] = Bmatrix[0]
                else:
                    Bd[0, ::n_dof] = Bmatrix[0]
                    Bd[3, ::n_dof] = Bmatrix[1]
                    Bd[5, ::n_dof] = Bmatrix[2]
                    Bd[1, 1::n_dof] = Bmatrix[1]
                    Bd[3, 1::n_dof] = Bmatrix[0]
                    Bd[4, 1::n_dof] = Bmatrix[2]
                    Bd[2, 2::n_dof] = Bmatrix[2]
                    Bd[4, 2::n_dof] = Bmatrix[1]
                    Bd[5, 2::n_dof] = Bmatrix[0]

                self._cnst[mater_id].calc_stress_elastic(
                    u_disp=u_elm,
                    bm=Bd,
                    itg=self._mesh.itg_idx(elm=ielm, itg=itg),
                    plane_stress_type=self._config["plane_stress"],
                )
