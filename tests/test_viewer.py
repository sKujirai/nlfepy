import os
import sys
import numpy as np
import logging
from logging import getLogger
import matplotlib.pyplot as plt
# import dmsh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy import Viewer
from nlfepy.io import VtuReader


def main():

    vtu_path = 'result.vtu'

    reader = VtuReader()
    reader.read(vtu_path)
    stress = reader.get_elm_value('stress', sys=[0, 1, 3])
    # print(stress.shape, stress.ndim)

    viewer_cnf = [
        {'val': 'stress', 'sys': [0, 1, 3], 'plot': 'fill'},
        {'val': 'u_disp', 'plot': 'contour'},
    ]
    projection = '3d' if reader.mesh['n_dof'] == 3 else '2d'
    viewer = Viewer(projection=projection)
    viewer.multi_plot(vtu_path, viewer_cnf)
    viewer.show()

    print('OK')


if __name__ == '__main__':
    main()
