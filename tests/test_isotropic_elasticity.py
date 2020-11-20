import os
import sys
import numpy as np
import logging
from logging import getLogger
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy.util import get_shape_function
from nlfepy.io import VtuReader, Viewer2d, Viewer3d


def main(mesh_path):

    logging.basicConfig(level=logging.INFO)
    logger = getLogger('isotropic')
    logger.info('Program start...')

    # Read mesh
    logger.info('Reading mesh...')
    reader = VtuReader(mesh_path)
    n_dof = reader.n_dof
    coords = reader.coords
    connectivity = reader.connectivity
    n_point = reader.n_point
    n_element = reader.n_element
    n_node = reader.connectivity.shape[1]
    if n_dof == 2:
        n_dfdof = 3
    else:
        n_dfdof = 6

    # Shape function
    shapef = get_shape_function(n_dof=n_dof, n_node=n_node)

    # Boundary condition (penalty method)
    BC = reader.bc
    PENALTY_COEFFICIENT = 1.e8

    # Traction vector
    Traction = BC['traction']

    # Elastic modulus matrix
    YOUNG = 70.3e9
    POISSON = 0.35
    C11 = YOUNG / ((1. + POISSON) * (1. - 2.*POISSON)) * (1. - POISSON)
    C12 = YOUNG / ((1. + POISSON) * (1. - 2.*POISSON)) * POISSON
    C44 = (C11 - C12) / 2.

    if n_dof == 2:
        Cmatrix = np.array([
            [C11, C12, 0.],
            [C12, C11, 0.],
            [0., 0., C44],
        ])
    else:
        Cmatrix = np.array([
            [C11, C12, C12, 0., 0., 0.],
            [C12, C11, C12, 0., 0., 0.],
            [C12, C12, C11, 0., 0., 0.],
            [0., 0., 0., C44, 0., 0.],
            [0., 0., 0., 0., C44, 0.],
            [0., 0., 0., 0., 0., C44]
        ])

    # Make global stiffness matrix & global force vector
    logger.info('Analysis start...')
    Kmatrix = np.zeros((n_dof * n_point, n_dof * n_point))
    Fvector = np.zeros(n_dof * n_point)
    for ielm in range(n_element):
        cod = coords[:, connectivity[ielm]][:n_dof]
        ke = np.zeros((n_dof * n_node, n_dof * n_node))
        fe = np.zeros(n_dof * n_node)
        Nmatrix = shapef['area'].Shpfnc

        # Make element stiffness matrix
        for iint in range(shapef['vol'].n_intgp):
            Bmatrix, wdetJv = shapef['vol'].get_bmatrix(cod, iint)
            Bd = np.zeros((n_dfdof, n_dof * n_node))
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
            ke += np.dot(Bd.T, np.dot(Cmatrix, Bd)) * wdetJv
            fe_t = 0.
            fe_b = 0.
            fe += fe_t + fe_b

        # Apply traction
        for ifce in range(shapef['vol'].n_face):
            codf = coords[:, connectivity[ielm][shapef['vol'].idx_face[ifce]]][:n_dof]
            trc = Traction[:, connectivity[ielm][shapef['vol'].idx_face[ifce]]][:n_dof].T.flatten()
            for jint in range(shapef['area'].n_intgp):
                wdetJa = shapef['area'].get_wdetJ(codf, jint)
                Na = np.zeros((n_dof, n_dof * shapef['area'].n_node))
                if n_dof == 2:
                    Na[0, ::n_dof] = Nmatrix[:, jint]
                    Na[1, 1::n_dof] = Nmatrix[:, jint]
                else:
                    Na[0, ::n_dof] = Nmatrix[:, jint]
                    Na[1, 1::n_dof] = Nmatrix[:, jint]
                    Na[2, 2::n_dof] = Nmatrix[:, jint]
                fe_t = np.dot(Na.T, np.dot(Na, trc)) * wdetJa
                for i, inod in enumerate(shapef['vol'].idx_face[ifce]):
                    fe[n_dof*inod:n_dof*(inod + 1)] += fe_t[n_dof*i:n_dof*(i + 1)]

        # Assemble global stiffness and force
        for inod in range(n_node):
            for jnod in range(n_node):
                for idof in range(n_dof):
                    for jdof in range(n_dof):
                        ipnt = connectivity[ielm, inod]
                        jpnt = connectivity[ielm, jnod]
                        Kmatrix[n_dof * ipnt + idof, n_dof * jpnt + jdof] += ke[n_dof * inod + idof, n_dof * jnod + jdof]
            for idof in range(n_dof):
                Fvector[n_dof * connectivity[ielm, inod] + idof] += fe[n_dof * inod + idof]

    # Handle boundary condition
    penalty = PENALTY_COEFFICIENT * np.max(np.abs(Kmatrix))
    for idx in BC['idx_fix']:
        Kmatrix[idx, idx] += penalty
    if BC['type'] == 'displacement':
        for i, idx in enumerate(BC['idx_disp']):
            Fvector[idx] += penalty * BC['displacement'][i]
    # elif BC['type'] == 'load':
    #     for i, idx in enumerate(BC['idx_disp']):
    #         Fvector[idx] += BC['Load'][i]

    # Solve KU=F
    logger.info('Solving KU=F...')
    Kmatrix = csr_matrix(Kmatrix)
    Uvector = spsolve(Kmatrix, Fvector, use_umfpack=True)

    # Update coordinates
    coords[0] += Uvector[::n_dof]
    coords[1] += Uvector[1::n_dof]
    if n_dof == 3:
        coords[2] += Uvector[2::n_dof]

    # Plot result
    logger.info('Drawing mesh...')
    if n_dof == 2:
        viewer = Viewer2d()
        viewer.set(coords, connectivity)
    else:
        viewer = Viewer3d()
        viewer.set(coords, connectivity, shapef)
    # viewer.save('result.png')
    viewer.show()

    logger.info('Program end')


if __name__ == '__main__':
    # main('tests/data/mesh.vtu')
    main('tests/data/mesh_3d.vtu')
