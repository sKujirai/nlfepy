import os
import sys
import numpy as np
import logging
from logging import getLogger
import matplotlib.pyplot as plt
# import dmsh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy import Mesh, Material, Variable, Constitutive, Viewer
from nlfepy.io import VtuReader


def main():

    vtu_path = 'result.vtu'

    reader = VtuReader()
    reader.read(vtu_path)
    stress = reader.get_elm_value('stress', sys=[0, 1, 3])
    print(stress.shape, stress.ndim)

    print('OK')


if __name__ == '__main__':
    main()
