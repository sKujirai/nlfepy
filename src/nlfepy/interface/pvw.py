import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .integral_equation import IntegralEquation


class PVW(IntegralEquation):
    """
    Principle of virtual work inheriting class: IntegralEquation

    Attributes
    ----------
    mesh :
        Mesh class (See mesh.py)
    cnst :
        Constitutive equation class
    val :
        Variables (physical quantity) (See variable.py)
    params : dict
        Parameters
    """

    def __init__(self, *, mesh, cnst, val=None, params: dict = {}) -> None:

        super().__init__(mesh=mesh, cnst=cnst, val=val, params=params)

    def solve(self) -> None:
        """
        Solve the governing equation of the deformation field.
        """

        n_dof = self._mesh.n_dof
        n_dfdof = self._mesh.n_dfdof
        n_point = self._mesh.n_point
        n_element = self._mesh.n_element
        connectivity = self._mesh.connectivity

        BC = self._mesh.bc
        Traction = BC["traction"] if "traction" in BC else None
        BodyForce = BC["body_force"] if "body_force" in BC else None

        Kmatrix = np.zeros((n_dof * n_point, n_dof * n_point))
        Fvector = np.zeros(n_dof * n_point)

        # Make global stiffness matrix & global force vector
        self._logger.info("Making stiffness matrix")
        for ielm in range(n_element):

            mater_id = self._mesh.material_numbers[ielm]
            n_node_v = self._mesh.n_node("vol", elm=ielm)
            n_intgp_v = self._mesh.n_intgp("vol", elm=ielm)

            # Make element stiffness matrix
            ke = np.zeros((n_dof * n_node_v, n_dof * n_node_v))
            fe = np.zeros(n_dof * n_node_v)

            Bmatrix, wdetJv = self._mesh.get_Bmatrix("vol", elm=ielm)

            # Plane stress condition
            if n_dof == 2 and self._config["plane_stress"] > 0:
                wdetJv *= self._cnst[mater_id].get_thickness[
                    self._mesh.itg_idx(elm=ielm)
                ]

            # Element stiffness
            Bd = np.zeros((n_intgp_v, n_dfdof, n_dof * n_node_v))
            Ce = np.zeros((n_intgp_v, n_dfdof, n_dfdof))
            if n_dof == 2:
                Bd[:, 0, ::n_dof] = Bmatrix[:, 0]
                Bd[:, 2, ::n_dof] = Bmatrix[:, 1]
                Bd[:, 1, 1::n_dof] = Bmatrix[:, 1]
                Bd[:, 2, 1::n_dof] = Bmatrix[:, 0]
            else:
                Bd[:, 0, ::n_dof] = Bmatrix[:, 0]
                Bd[:, 3, ::n_dof] = Bmatrix[:, 1]
                Bd[:, 5, ::n_dof] = Bmatrix[:, 2]
                Bd[:, 1, 1::n_dof] = Bmatrix[:, 1]
                Bd[:, 3, 1::n_dof] = Bmatrix[:, 0]
                Bd[:, 4, 1::n_dof] = Bmatrix[:, 2]
                Bd[:, 2, 2::n_dof] = Bmatrix[:, 2]
                Bd[:, 4, 2::n_dof] = Bmatrix[:, 1]
                Bd[:, 5, 2::n_dof] = Bmatrix[:, 0]
            Ce = self._cnst[mater_id].get_elastic_modulus(
                n_dof, self._config["plane_stress"]
            )
            ke = np.einsum("bki, kl, blj, b -> ij", Bd, Ce, Bd, wdetJv)

            # Body force
            if BodyForce is not None:
                bf = BodyForce[np.array(connectivity[ielm])][:, :n_dof].flatten()
                Nbmatrix = self._mesh.get_shpfnc("vol", elm=ielm)
                Nb = np.zeros((n_intgp_v, n_dof, n_dof * n_node_v))
                if n_dof == 2:
                    Nb[:, 0, ::n_dof] = Nbmatrix
                    Nb[:, 1, 1::n_dof] = Nbmatrix
                else:
                    Nb[:, 0, ::n_dof] = Nbmatrix
                    Nb[:, 1, 1::n_dof] = Nbmatrix
                    Nb[:, 2, 2::n_dof] = Nbmatrix
                fe_b = np.einsum("bji, bjk, k, b -> i", Nb, Nb, bf, wdetJv)
                fe += fe_b

            # Apply traction
            if Traction is not None:
                for idx_nd in self._mesh.idx_face("vol", elm=ielm):
                    trc = Traction[np.array(connectivity[ielm])[idx_nd]][
                        :, :n_dof
                    ].flatten()
                    n_node_a = self._mesh.n_node("area", elm=ielm)
                    n_intgp_a = self._mesh.n_intgp("area", elm=ielm)
                    Namatrix, wdetJa = self._mesh.get_Nmatrix(
                        "area", elm=ielm, nds=idx_nd
                    )
                    Na = np.zeros((n_intgp_a, n_dof, n_dof * n_node_a))
                    if n_dof == 2:
                        Na[:, 0, ::n_dof] = Namatrix
                        Na[:, 1, 1::n_dof] = Namatrix
                    else:
                        Na[:, 0, ::n_dof] = Namatrix
                        Na[:, 1, 1::n_dof] = Namatrix
                        Na[:, 2, 2::n_dof] = Namatrix
                    fe_t = np.einsum("bji, bjk, k, b -> i", Na, Na, trc, wdetJa)
                    for i, inod in enumerate(idx_nd):
                        fe[n_dof * inod : n_dof * (inod + 1)] += fe_t[
                            n_dof * i : n_dof * (i + 1)
                        ]

            # Assemble global stiffness and force
            for inod in range(n_node_v):
                for jnod in range(n_node_v):
                    for idof in range(n_dof):
                        for jdof in range(n_dof):
                            ipnt = connectivity[ielm][inod]
                            jpnt = connectivity[ielm][jnod]
                            Kmatrix[n_dof * ipnt + idof, n_dof * jpnt + jdof] += ke[
                                n_dof * inod + idof, n_dof * jnod + jdof
                            ]
                for idof in range(n_dof):
                    Fvector[n_dof * connectivity[ielm][inod] + idof] += fe[
                        n_dof * inod + idof
                    ]

        # Handle boundary condition
        self._logger.info("Handling B.C.")
        penalty = self._config["penalty_coefficient"] * np.max(np.abs(Kmatrix))
        for idx in BC["idx_fix"]:
            Kmatrix[idx, idx] += penalty
        if "idx_disp" in BC:
            for i, idx in enumerate(BC["idx_disp"]):
                Fvector[idx] += penalty * BC["displacement"][i]
        if "applied_force" in BC:
            Fvector += BC["applied_force"][:, :n_dof].flatten()

        # Solve KU=F
        self._logger.info("Solving KU=F")
        Kmatrix = csr_matrix(Kmatrix)
        Uvector = spsolve(Kmatrix, Fvector, use_umfpack=True)

        # Update coordinates
        self._logger.info("Updating global coordinates")
        for idof in range(n_dof):
            self._val["u_disp"][idof] = Uvector[idof::n_dof]
            self._mesh.coords[idof] += Uvector[idof::n_dof]
