import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nlfepy.shape import (
    Nd2,
    Nd3,
    Nd4,
    Tri3,
    Tri6,
    Tri10,
    Quad4,
    Quad8,
    Quad9,
    Quad12,
    Quad16,
    Tet4,
    Hexa8,
)


def main():

    nd2 = Nd2()
    nd3 = Nd3()
    nd4 = Nd4()
    tri3 = Tri3()
    tri6 = Tri6()
    tri10 = Tri10()
    quad4 = Quad4()
    quad8 = Quad8()
    quad9 = Quad9()
    quad12 = Quad12()
    quad16 = Quad16()
    tet4 = Tet4()
    hexa8 = Hexa8()

    print(nd2.name)
    print(nd3.name)
    print(nd4.name)
    print(tri3.name)
    print(tri6.name)
    print(tri10.name)
    print(quad4.name)
    print(quad8.name)
    print(quad9.name)
    print(quad12.name)
    print(quad16.name)
    print(tet4.name)
    print(hexa8.name)


if __name__ == "__main__":
    main()
