import os
import sys
import numpy as np
import logging
from logging import getLogger
# import dmsh
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

    # Prepare mesh using dmsh
    # geo = dmsh.Rectangle(0., 1., 0., 1.)
    # coords, connectivity = dmsh.generate(geo, 0.1)

    # # Mesh class
    # mesh = Mesh()
    # mesh.set_shape(coords=coords.T, connectivity=connectivity)
    # mesh.set_bc(constraint='compression', value=0.001)

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
        'val': 'rand',
    }
    viewer = Viewer(mesh=mesh)
    viewer.show(show_cbar=False)
    val = {}
    val['rand'] = np.random.rand(mesh.n_element)
    viewer.set(values=val, params=viewer_params)
    # viewer.save('result.png')
    viewer.show()

    logger.info('Program end')


if __name__ == '__main__':
    # main('tests/data/mesh.vtu')
    # main('tests/data/mesh_load.vtu')
    main('tests/data/mesh_3d.vtu')
