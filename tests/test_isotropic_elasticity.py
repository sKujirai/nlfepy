import os
from os import read
import sys
import numpy as np
import logging
from logging import getLogger
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy.mesh import Mesh
from nlfepy.material import get_material_list
from nlfepy.interface import PVW
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

    # Set material
    logger.info('Setting material...')
    mater = get_material_list(['Al'])

    # Solve the governing equation (Principle of virtual work)
    logger.info('Solving the governing equation...')
    pvw_params = {
        'logging': True,
    }
    pvw = PVW(mesh=mesh, mater=mater, params=pvw_params)
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
