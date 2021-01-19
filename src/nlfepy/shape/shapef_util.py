from ..shape import Nd2, Tri3, Tri6, Quad4, Hexa8
import sys
from logging import getLogger
from .shape_function import ShapeFunction
from typing import Dict


def get_element_name(*, n_dof: int, n_node: int) -> str:
    """
    Get finite element name

    Parameters
    ----------
    n_dof : int
        Number of degrees of freedom
    n_node : int
        Number of nodes of each element

    Returns
    -------
    element_name : str
        Name of finite element
    """

    logger = getLogger("shape")

    element_name = None

    if n_dof == 2:
        if n_node == 3:
            element_name = "TRI3"
        elif n_node == 4:
            element_name = "QUAD4"
        elif n_node == 6:
            element_name = "TRI6"
        elif n_node == 8:
            element_name = "QUAD8"
        else:
            logger.error("Invalid 2D element")
            sys.exit(1)
    else:
        if n_node == 4:
            element_name = "TET4"
        elif n_node == 8:
            element_name = "HEXA8"
        else:
            logger.error("Invalid 3D element")
            sys.exit(1)

    return element_name


def get_shape_function(element_name: str) -> Dict[str, ShapeFunction]:
    """
    Get shape function

    Parameters
    ----------
    element_name : str
        Name of finite element

    Returns
    -------
    shapef : dict
        Dictionary of shape function ('vol', 'area')
    """

    logger = getLogger("shape")

    shapef: Dict[str, ShapeFunction] = {}

    if element_name == "TRI3":
        shapef["vol"] = Tri3()
        shapef["area"] = Nd2()
    elif element_name == "TRI6":
        shapef["vol"] = Tri6()
        shapef["area"] = Nd2()
    elif element_name == "QUAD4":
        shapef["vol"] = Quad4()
        shapef["area"] = Nd2()
    elif element_name == "HEXA8":
        shapef["vol"] = Hexa8()
        shapef["area"] = Quad4()
    else:
        logger.error("Invalid element name: {}".format(element_name))
        sys.exit(1)

    return shapef
