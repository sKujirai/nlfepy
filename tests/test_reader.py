import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy.io import VtuReader


def main(mesh_path):
    reader = VtuReader(mesh_path)
    print(reader.coords.shape)
    print(reader.coords)
    print(reader.connectivity.shape)
    print(reader.connectivity)
    print(reader.bc)
    print('OK')


if __name__ == '__main__':
    main('tests/data/mesh_mpc.vtu')
