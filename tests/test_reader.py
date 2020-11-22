import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from nlfepy.io import VtuReader


def main(mesh_path):
    reader = VtuReader(mesh_path)
    mesh = reader.mesh
    BC = reader.bc

    for key, value in mesh.items():
        print(key)
        print(value)

    for key, value in BC.items():
        print(key, value)

    print('OK')


if __name__ == '__main__':
    main('tests/data/mesh_mpc.vtu')
