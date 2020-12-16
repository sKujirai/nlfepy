import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .integral_equation import IntegralEquation


class PVW_UL(IntegralEquation):
    """
    Principle of virtual work (the updated Lagrangian formulation) inheriting class: IntegralEquatison

    Attributes
    ----------
    mesh :
        Mesh class (See mesh.py)
    cnst :
        Constitutive equation class
    val :
        Variables (physical quantity)  (See variable.py)
    params : dict
        Parameters
    deltaU : ndarray
        Incremental displacement
    Fint : ndarray
        Internal force
    Fext : ndarray
        External force
    Frsd : ndarray
        Residual force
    """

    def __init__(self, *, mesh, cnst, val=None, params: dict = {}) -> None:

        super().__init__(mesh=mesh, cnst=cnst, val=val, params=params)

        n_dof = self._mesh.n_dof
        n_point = self._mesh.n_point

        self.Fint = np.zeros((n_dof * n_point))
        self.Fext = np.zeros((n_dof * n_point))
        self.Frsd = np.zeros((n_dof * n_point))

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
        TractionRate = BC["traction"] if "traction" in BC else None
        BodyForceRate = BC["body_force"] if "body_force" in BC else None

        Kmatrix = np.zeros((n_dof * n_point, n_dof * n_point))
        Fvector = np.zeros(n_dof * n_point)
        dFext = np.zeros(n_dof * n_point)
        dFapp = np.zeros(n_dof * n_point)
        tFint = np.zeros(n_dof * n_point)

        # Make global stiffness matrix & global force vector
        self._logger.info("Making stiffness matrix")
        for ielm in range(n_element):

            mater_id = self._mesh.material_numbers[ielm]
            n_node_v = self._mesh.n_node("vol", elm=ielm)
            n_intgp_v = self._mesh.n_intgp("vol", elm=ielm)

            Nbmatrix = self._mesh.get_shpfnc("vol", elm=ielm)

            dUelm = self._val["deltau"][:, np.array(connectivity[ielm])]

            # Make element stiffness matrix
            ke = np.zeros((n_dof * n_node_v, n_dof * n_node_v))
            fe_int = np.zeros(n_dof * n_node_v)
            dfe_app = np.zeros(n_dof * n_node_v)
            dfe_ext = np.zeros(n_dof * n_node_v)

            for itg in range(n_intgp_v):

                # Bmatrix & wdetJ
                Bmatrix, wdetJv = self._mesh.get_Bmatrix("vol", elm=ielm, itg=itg)

                # Plane stress condition
                if n_dof == 2 and self._config["plane_stress"] > 0:
                    wdetJv *= self._cnst[mater_id].get_thickness[
                        self._mesh.itg_idx(elm=ielm, itg=itg)
                    ]

                # [C], {R}, {T}
                Cmatrix, Rvector, Tvector = self._cnst[mater_id].constitutive_equation(
                    du=dUelm,
                    bm=Bmatrix,
                    itg=self._mesh.itg_idx(elm=ielm, itg=itg),
                    plane_stress_type=self._config["plane_stress"],
                )
                Tmatrix = Tvector[[0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(3, 3)

                # [ke], {dfint}, {Fint}, {dfext_b}, {fext_b}
                Bd = np.zeros((n_dfdof, n_dof * n_node_v))
                Bl = np.zeros((n_dof * n_dof, n_dof * n_node_v))
                Td = np.zeros((n_dfdof, n_dfdof))
                Tl = np.zeros((n_dof * n_dof, n_dof * n_dof))
                TtrL = np.zeros((n_dof * n_dof, n_dof * n_dof))
                if n_dof == 2:
                    # [Bd]
                    Bd[0, ::n_dof] = Bmatrix[0]
                    Bd[2, ::n_dof] = Bmatrix[1]
                    Bd[1, 1::n_dof] = Bmatrix[1]
                    Bd[2, 1::n_dof] = Bmatrix[0]
                    # [Bl]
                    Bl[0, ::n_dof] = Bmatrix[0]
                    Bl[1, 1::n_dof] = Bmatrix[0]
                    Bl[2, ::n_dof] = Bmatrix[1]
                    Bl[3, 1::n_dof] = Bmatrix[1]
                    # [Td]
                    Td[0, 0] = 2.0 * Tmatrix[0, 0]
                    Td[1, 1] = 2.0 * Tmatrix[1, 1]
                    Td[2, 2] = 0.5 * (Tmatrix[0, 0] + Tmatrix[1, 1])
                    Td[0, 2] = Tmatrix[0, 1]
                    Td[2, 0] = Tmatrix[0, 1]
                    Td[1, 2] = Tmatrix[0, 1]
                    Td[2, 1] = Tmatrix[0, 1]
                    # [Tl]
                    Tl[::n_dof, :][:, ::n_dof] = Tmatrix[:2, :2]
                    Tl[1::n_dof, :][:, 1::n_dof] = Tmatrix[:2, :2]
                    # [TtrL]
                    TtrL[0:, :] = Tmatrix[:2, :2].flatten()
                    TtrL[3:, :] = Tmatrix[:2, :2].flatten()
                    # {T}
                    Tvector = Tvector[[0, 1, 3]]
                    # [C] & {R}
                    Cmatrix, Rvector = self._cnst[
                        mater_id
                    ].calc_correction_term_plane_stress_CR(
                        Cmatrix, Rvector, self._config["plane_stress"]
                    )
                else:
                    # [Bd]
                    Bd[0, ::n_dof] = Bmatrix[0]
                    Bd[3, ::n_dof] = Bmatrix[1]
                    Bd[5, ::n_dof] = Bmatrix[2]
                    Bd[1, 1::n_dof] = Bmatrix[1]
                    Bd[3, 1::n_dof] = Bmatrix[0]
                    Bd[4, 1::n_dof] = Bmatrix[2]
                    Bd[2, 2::n_dof] = Bmatrix[2]
                    Bd[4, 2::n_dof] = Bmatrix[1]
                    Bd[5, 2::n_dof] = Bmatrix[0]
                    # [Bl]
                    Bl[0, ::n_dof] = Bmatrix[0]
                    Bl[1, 1::n_dof] = Bmatrix[0]
                    Bl[2, 2::n_dof] = Bmatrix[0]
                    Bl[3, ::n_dof] = Bmatrix[1]
                    Bl[4, 1::n_dof] = Bmatrix[1]
                    Bl[5, 2::n_dof] = Bmatrix[1]
                    Bl[6, ::n_dof] = Bmatrix[2]
                    Bl[7, 1::n_dof] = Bmatrix[2]
                    Bl[8, 2::n_dof] = Bmatrix[2]
                    # [Td]
                    Td[0, 0] = 2.0 * Tmatrix[0, 0]
                    Td[1, 1] = 2.0 * Tmatrix[1, 1]
                    Td[2, 2] = 2.0 * Tmatrix[2, 2]
                    Td[3, 3] = 0.5 * (Tmatrix[0, 0] + Tmatrix[1, 1])
                    Td[4, 4] = 0.5 * (Tmatrix[1, 1] + Tmatrix[2, 2])
                    Td[5, 5] = 0.5 * (Tmatrix[2, 2] + Tmatrix[0, 0])
                    Td[0, 3] = Tmatrix[0, 1]
                    Td[0, 5] = Tmatrix[2, 0]
                    Td[1, 3] = Tmatrix[0, 1]
                    Td[1, 4] = Tmatrix[1, 2]
                    Td[2, 4] = Tmatrix[1, 2]
                    Td[2, 5] = Tmatrix[2, 0]
                    Td[3, 0] = Tmatrix[0, 1]
                    Td[3, 1] = Tmatrix[0, 1]
                    Td[4, 1] = Tmatrix[1, 2]
                    Td[4, 2] = Tmatrix[1, 2]
                    Td[5, 0] = Tmatrix[2, 0]
                    Td[5, 2] = Tmatrix[2, 0]
                    Td[3, 4] = 0.5 * Tmatrix[2, 0]
                    Td[3, 5] = 0.5 * Tmatrix[1, 2]
                    Td[4, 3] = 0.5 * Tmatrix[2, 0]
                    Td[4, 5] = 0.5 * Tmatrix[0, 1]
                    Td[5, 3] = 0.5 * Tmatrix[1, 2]
                    Td[5, 4] = 0.5 * Tmatrix[0, 1]
                    # [Tl]
                    Tl[::n_dof, :][:, ::n_dof] = Tmatrix
                    Tl[1::n_dof, :][:, 1::n_dof] = Tmatrix
                    Tl[2::n_dof, :][:, 2::n_dof] = Tmatrix
                    # [TtrL]
                    TtrL[0:, :] = Tmatrix.flatten()
                    TtrL[4:, :] = Tmatrix.flatten()
                    TtrL[8:, :] = Tmatrix.flatten()

                # [ke] += [Bd]^T[D][Bd]*wdetJ
                ke += np.dot(Bd.T, np.dot(Cmatrix, Bd)) * wdetJv
                # {dfapp}
                dfe_app += np.dot(Bd.T, Rvector) * wdetJv
                # {Fint}
                fe_int += np.dot(Bd.T, Tvector) * wdetJv
                # Body force
                if BodyForceRate is not None:
                    bfr = BodyForceRate[np.array(connectivity[ielm])][
                        :, :n_dof
                    ].flatten()
                    Nb = np.zeros((n_dof, n_dof * n_node_v))
                    if n_dof == 2:
                        Nb[0, ::n_dof] = Nbmatrix[:, itg]
                        Nb[1, 1::n_dof] = Nbmatrix[:, itg]
                    else:
                        Nb[0, ::n_dof] = Nbmatrix[:, itg]
                        Nb[1, 1::n_dof] = Nbmatrix[:, itg]
                        Nb[2, 2::n_dof] = Nbmatrix[:, itg]
                    # {dfext_b}
                    dfe_ext += np.dot(Nb.T, np.dot(Nb, bfr)) * wdetJv

            # Apply traction
            if TractionRate is not None:
                for idx_nd in self._mesh.idx_face("vol", elm=ielm):
                    trcr = TractionRate[np.array(connectivity[ielm])[idx_nd]][
                        :, :n_dof
                    ].flatten()
                    n_node_a = self._mesh.n_node("area", elm=ielm)
                    n_intgp_a = self._mesh.n_intgp("area", elm=ielm)
                    for jtg in range(n_intgp_a):
                        Namatrix, wdetJa = self._mesh.get_Nmatrix(
                            "area", elm=ielm, itg=jtg, nds=idx_nd
                        )
                        Na = np.zeros((n_dof, n_dof * n_node_a))
                        if n_dof == 2:
                            Na[0, ::n_dof] = Namatrix[:, jtg]
                            Na[1, 1::n_dof] = Namatrix[:, jtg]
                        else:
                            Na[0, ::n_dof] = Namatrix[:, jtg]
                            Na[1, 1::n_dof] = Namatrix[:, jtg]
                            Na[2, 2::n_dof] = Namatrix[:, jtg]
                        dfe_t = np.dot(Na.T, np.dot(Na, trcr)) * wdetJa
                        for i, inod in enumerate(idx_nd):
                            dfe_ext[n_dof * inod : n_dof * (inod + 1)] += dfe_t[
                                n_dof * i : n_dof * (i + 1)
                            ]

            # Assemble
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
                    dFext[n_dof * connectivity[ielm][inod] + idof] += dfe_ext[
                        n_dof * inod + idof
                    ]
                    dFapp[n_dof * connectivity[ielm][inod] + idof] += dfe_app[
                        n_dof * inod + idof
                    ]
                    tFint[n_dof * connectivity[ielm][inod] + idof] += fe_int[
                        n_dof * inod + idof
                    ]

        # Set global force
        Fvector = dFext + dFapp + self.Frsd
        self.Fext += dFext
        self.Fint = tFint
        self.Frsd = self.Fext - self.Fint

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
            self._val["deltau"][idof] = Uvector[idof::n_dof]
            self._val["u_disp"][idof] += Uvector[idof::n_dof]
            self._mesh.coords[idof] += Uvector[idof::n_dof]
