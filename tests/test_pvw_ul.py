import os
import sys
import numpy as np
import logging
from logging import getLogger
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy.mesh import Mesh
from nlfepy.constitutive import get_constitutive_list
from nlfepy.interface import PVW_UL
from nlfepy.io import Viewer


def main(mesh_path):

    # Set logger
    logging.basicConfig(level=logging.INFO)
    logger = getLogger('pvw')
    logger.info('Program start...')

    # Read mesh
    logger.info('Setting mesh info...')
    mesh = Mesh()
    mesh.read(mesh_path)

    # Physical quantities
    # val = {}

    # Set constitutive
    logger.info('Setting constitutive equation...')
    # cnst_params = {}
    cnst_dict = {
        'Al': ['isotropic'],  # , cnst_params],
    }
    constitutive = get_constitutive_list(cnst_dict)  # , val)

    # Solve the governing equation (Principle of virtual work)
    logger.info('Solving the governing equation...')
    pvw_params = {
        'logging': True,
    }
    pvw = PVW_UL(
        mesh=mesh,
        cnst=constitutive,
        # val=val,
        params=pvw_params
    )
    pvw.solve()

    # Plot result
    logger.info('Drawing mesh...')
    viewer_params = {
        'cmap': 'rainbow',
        'lw': 1,
    }
    viewer = Viewer(mesh=mesh, params=viewer_params)
    val = None
    # val = np.random.rand(mesh.n_element)
    viewer.set(value=val)
    # viewer.save('result.png')
    viewer.show()

    logger.info('Program end')


if __name__ == '__main__':
    # main('tests/data/mesh.vtu')
    main('tests/data/mesh_3d.vtu')
