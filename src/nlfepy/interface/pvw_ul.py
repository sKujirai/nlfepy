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
        Variables (physical quantity) class
    params : dict
        Parameters
    """

    def __init__(self, *, mesh, cnst, val=None, params: dict = {}) -> None:

        super().__init__(mesh=mesh, val=val, params=params)

        self.cnst = cnst

        n_dof = self.mesh.n_dof
        n_point = self.mesh.n_point
        self.deltaU = np.zeros((n_dof, n_point))
        self.Traction = np.zeros((n_dof, n_point))
        self.BodyForce = np.zeros((n_dof, n_point))

    def solve(self) -> None:
        """
        Solve the governing equation of the deformation field.
        """

        n_dof = self.mesh.n_dof
        n_dfdof = self.mesh.n_dfdof
        n_point = self.mesh.n_point
        n_element = self.mesh.n_element
        connectivity = self.mesh.connectivity

        BC = self.mesh.bc
        TractionRate = None
        BodyForceRate = None
        if 'traction' in BC:
            TractionRate = BC['traction']
            self.Traction += TractionRate
        if 'body_force' in BC:
            BodyForceRate = BC['body_force']
            self.BodyForce += BodyForceRate

        Kmatrix = np.zeros((n_dof * n_point, n_dof * n_point))
        Fvector = np.zeros(n_dof * n_point)

        # Make global stiffness matrix & global force vector
        self.logger.info('Making stiffness matrix')
        for ielm in range(n_element):

            mater_id = self.mesh.material_numbers[ielm]
            n_node_v = self.mesh.n_node('vol', elm=ielm)
            n_intgp_v = self.mesh.n_intgp('vol', elm=ielm)

            Nbmatrix = self.mesh.get_Shpfnc('vol', elm=ielm)

            dUelm = self.deltaU[:, np.array(connectivity[ielm])]
            # dUelm = np.zeros((n_dof, n_node_v))
            # for idof in range(n_dof):
            #     dUelm[idof] = self.deltaU[n_dof * np.array(connectivity[ielm]) + idof]

            # Make element stiffness matrix
            ke = np.zeros((n_dof * n_node_v, n_dof * n_node_v))
            fe = np.zeros(n_dof * n_node_v)
            fe_int = np.zeros(n_dof * n_node_v)
            fe_ext = np.zeros(n_dof * n_node_v)
            dfe_int = np.zeros(n_dof * n_node_v)
            dfe_ext = np.zeros(n_dof * n_node_v)

            for itg in range(n_intgp_v):

                # Bmatrix & wdetJ
                Bmatrix, wdetJv = self.mesh.get_Bmatrix('vol', elm=ielm, itg=itg)

                # [C], {R}, {T}
                Cmatrix, Rmatrix, Tmatrix = self.cnst[mater_id].constitutive_equation(du=dUelm, bm=Bmatrix, itg=itg)

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
                    Td[0, 0] = 2.*Tmatrix[0, 0]
                    Td[1, 1] = 2.*Tmatrix[1, 1]
                    Td[2, 2] = 0.5*(Tmatrix[0, 0] + Tmatrix[1, 1])
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
                    Tvector = Tmatrix.flatten()[[0, 4, 1]]
                    # [R]
                    Rvector = Rmatrix.flatten()[[0, 4, 1]]
                    # [C]
                    Cmatrix = Cmatrix[[0, 1, 3], :][:, [0, 1, 3]]
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
                    Td[0, 0] = 2.*Tmatrix[0, 0]
                    Td[1, 1] = 2.*Tmatrix[1, 1]
                    Td[2, 2] = 2.*Tmatrix[2, 2]
                    Td[3, 3] = 0.5*(Tmatrix[0, 0] + Tmatrix[1, 1])
                    Td[4, 4] = 0.5*(Tmatrix[1, 1] + Tmatrix[2, 2])
                    Td[5, 5] = 0.5*(Tmatrix[2, 2] + Tmatrix[0, 0])
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
                    Td[3, 4] = 0.5*Tmatrix[2, 0]
                    Td[3, 5] = 0.5*Tmatrix[1, 2]
                    Td[4, 3] = 0.5*Tmatrix[2, 0]
                    Td[4, 5] = 0.5*Tmatrix[0, 1]
                    Td[5, 3] = 0.5*Tmatrix[1, 2]
                    Td[5, 4] = 0.5*Tmatrix[0, 1]
                    # [Tl]
                    Tl[::n_dof, :][:, ::n_dof] = Tmatrix
                    Tl[1::n_dof, :][:, 1::n_dof] = Tmatrix
                    Tl[2::n_dof, :][:, 2::n_dof] = Tmatrix
                    # [TtrL]
                    TtrL[0:, :] = Tmatrix.flatten()
                    TtrL[4:, :] = Tmatrix.flatten()
                    TtrL[8:, :] = Tmatrix.flatten()
                    # {T}
                    Tvector = Tmatrix.flatten()[[0, 4, 8, 1, 5, 6]]
                    # {R}
                    Rvector = Rmatrix.flatten()[[0, 4, 8, 1, 5, 6]]

                # [ke] += [Bd]^T[D][Bd]*wdetJ
                ke += np.dot(Bd.T, np.dot(Cmatrix, Bd)) * wdetJv
                # {dfint}
                dfe_int += np.dot(Bd.T, Rvector) * wdetJv
                # {Fint}
                fe_int += np.dot(Bd.T, Tvector) * wdetJv
                # Body force
                if BodyForceRate is not None:
                    bfr = BodyForceRate[:, np.array(connectivity[ielm])][:n_dof].T.flatten()
                    bf = self.BodyForce[:, np.array(connectivity[ielm])][:n_dof].T.flatten()
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
                    # {fext_b}
                    fe_ext += np.dot(Nb.T, np.dot(Nb, bf)) * wdetJv

            # Apply traction
            if TractionRate is not None:
                for idx_nd in self.mesh.idx_face('vol', elm=ielm):
                    trcr = TractionRate[:, np.array(connectivity[ielm])[idx_nd]][:n_dof].T.flatten()
                    trc = self.Traction[:, np.array(connectivity[ielm])[idx_nd]][:n_dof].T.flatten()
                    n_node_a = self.mesh.n_node('area', elm=ielm)
                    n_intgp_a = self.mesh.n_intgp('area', elm=ielm)
                    for jtg in range(n_intgp_a):
                        Namatrix, wdetJa = self.mesh.get_Nmatrix('area', elm=ielm, itg=jtg, nds=idx_nd)
                        Na = np.zeros((n_dof, n_dof * n_node_a))
                        if n_dof == 2:
                            Na[0, ::n_dof] = Namatrix[:, jtg]
                            Na[1, 1::n_dof] = Namatrix[:, jtg]
                        else:
                            Na[0, ::n_dof] = Namatrix[:, jtg]
                            Na[1, 1::n_dof] = Namatrix[:, jtg]
                            Na[2, 2::n_dof] = Namatrix[:, jtg]
                        dfe_t = np.dot(Na.T, np.dot(Na, trcr)) * wdetJa
                        fe_t = np.dot(Na.T, np.dot(Na, trc)) * wdetJa
                        for i, inod in enumerate(idx_nd):
                            dfe_ext[n_dof*inod:n_dof*(inod + 1)] += dfe_t[n_dof*i:n_dof*(i + 1)]
                            fe_ext[n_dof*inod:n_dof*(inod + 1)] += fe_t[n_dof*i:n_dof*(i + 1)]

            fe = dfe_ext - dfe_int + fe_ext - fe_int

            # Assemble
            for inod in range(n_node_v):
                for jnod in range(n_node_v):
                    for idof in range(n_dof):
                        for jdof in range(n_dof):
                            ipnt = connectivity[ielm][inod]
                            jpnt = connectivity[ielm][jnod]
                            Kmatrix[n_dof * ipnt + idof, n_dof * jpnt + jdof] += ke[n_dof * inod + idof, n_dof * jnod + jdof]
                for idof in range(n_dof):
                    Fvector[n_dof * connectivity[ielm][inod] + idof] += fe[n_dof * inod + idof]

        # Handle boundary condition
        self.logger.info('Handling B.C.')
        penalty = self.config['penalty_coefficient'] * np.max(np.abs(Kmatrix))
        for idx in BC['idx_fix']:
            Kmatrix[idx, idx] += penalty
        if BC['type'] == 'displacement':
            for i, idx in enumerate(BC['idx_disp']):
                Fvector[idx] += penalty * BC['displacement'][i]

        # Solve KU=F
        self.logger.info('Solving KU=F')
        Kmatrix = csr_matrix(Kmatrix)
        Uvector = spsolve(Kmatrix, Fvector, use_umfpack=True)

        # Update coordinates
        self.logger.info('Updating global coordinates')
        for idof in range(n_dof):
            self.deltaU[idof] = Uvector[idof::n_dof]
            self.mesh.coords[idof] += Uvector[idof::n_dof]