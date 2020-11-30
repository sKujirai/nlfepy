import os
import sys
import numpy as np
import logging
from logging import getLogger
# import dmsh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy import Mesh, Material, Variable, Viewer
from nlfepy.interface import PVW
from nlfepy.io import VtuWriter


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

    # Mesh class
    # mesh = Mesh()
    # mesh.set_shape(coords=coords.T, connectivity=connectivity)
    # mesh.set_bc(constraint='compression', value=0.001)

    # Set material
    logger.info('Setting material...')
    mater = Material('Al')

    # Physical quantities
    vals = Variable()

    # Solve the governing equation (Principle of virtual work)
    logger.info('Solving the governing equation...')
    pvw_params = {
        'logging': True,
    }
    pvw = PVW(mesh=mesh, mater=mater, val=vals['point'], params=pvw_params)
    pvw.solve()

    # Save results
    logger.info('Saving results')
    writer = VtuWriter(mesh=mesh)
    writer.write('result.vtu')

    # Plot result
    logger.info('Drawing mesh...')
    projection = '3d' if mesh.n_dof == 3 else '2d'
    viewer = Viewer(projection=projection)

    # Check B.C.
    viewer.plot_bc(mesh)
    viewer.show()

    # Plot result
    vals['element']['rand'] = np.random.rand(mesh.n_element)
    viewer.plot(mesh=mesh, val=vals['element']['rand'])
    # viewer.save('result.png')
    viewer.show()

    # Contour plot
    if mesh.n_dof == 2:
        vals['point']['randp'] = np.random.rand(mesh.n_point)
        viewer.contour(mesh=mesh, val=vals['point']['randp'])
        viewer.show()

    logger.info('Program end')


if __name__ == '__main__':
    # main('tests/data/mesh.vtu')
    # main('tests/data/mesh_mpc.vtu')
    # main('tests/data/mesh_load.vtu')
    main('tests/data/mesh_3d.vtu')
    # main('tests/data/mesh_3d_load.vtu')
