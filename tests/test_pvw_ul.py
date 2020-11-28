import os
import sys
import numpy as np
import logging
from logging import getLogger
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy import Mesh, Variable, Constitutive, Viewer
from nlfepy.interface import PVW_UL


def main(mesh_path):

    # Number of steps
    n_steps = 10

    # Set logger
    logging.basicConfig(level=logging.INFO)
    logger = getLogger('pvw')
    logger.info('Program start...')

    # Read mesh
    logger.info('Setting mesh info...')
    mesh = Mesh()
    mesh.read(mesh_path)

    # Physical quantities
    vals = Variable()

    # Set constitutive
    logger.info('Setting constitutive equation...')
    # cnst_params = {}
    cnst_dict = {
        'Al': ['isotropic'],
        # 'Al': ['j2flow'],  # , cnst_params],
        # 'Al': ['crystal_plasticity'],  # , cnst_params],
    }
    constitutive = Constitutive(
        cnst_dict,
        nitg=mesh.n_tintgp,
        val=vals['itg']
    )

    # Solve the governing equation (Principle of virtual work)
    logger.info('Solving the governing equation...')
    pvw_params = {
        'logging': False,
    }
    pvw = PVW_UL(
        mesh=mesh,
        cnst=constitutive,
        val=vals['point'],
        params=pvw_params
    )

    # Main Loop
    for istep in range(n_steps):
        logger.info('Step No. {} / {}'.format(istep + 1, n_steps))
        pvw.solve()

    # Plot result
    logger.info('Drawing mesh...')
    viewer_params = {
        'cmap': 'rainbow',
        'lw': 1,
    }
    viewer = Viewer(mesh=mesh)
    val = {}
    viewer.set(values=val, params=viewer_params)
    # viewer.save('result.png')
    viewer.show()

    logger.info('Program end')


if __name__ == '__main__':
    # main('tests/data/mesh.vtu')
    main('tests/data/mesh_3d.vtu')
