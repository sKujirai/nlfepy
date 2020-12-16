import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nlfepy.shape import Nd2, Tri3, Tri6, Quad4, Hexa8


def main():

    nd2 = Nd2()
    tri3 = Tri3()
    tri6 = Tri6()
    quad4 = Quad4()
    hexa8 = Hexa8()

    print(nd2.name)
    print(tri3.name)
    print(tri6.name)
    print(quad4.name)
    print(hexa8.name)


if __name__ == "__main__":
    main()
