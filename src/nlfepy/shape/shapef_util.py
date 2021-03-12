from ..shape import (
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

    if n_dof == 1:
        if n_node == 2:
            element_name = "Nd2"
        elif n_node == 3:
            element_name = "Nd3"
        elif n_node == 4:
            element_name = "Nd4"
        else:
            logger.error("Invalid 1D element")
            sys.exit(1)
    elif n_dof == 2:
        if n_node == 3:
            element_name = "TRI3"
        elif n_node == 4:
            element_name = "QUAD4"
        elif n_node == 6:
            element_name = "TRI6"
        elif n_node == 8:
            element_name = "QUAD8"
        elif n_node == 9:
            element_name = "QUAD9"
        elif n_node == 10:
            element_name = "TRI10"
        elif n_node == 12:
            element_name = "QUAD12"
        elif n_node == 16:
            element_name = "QUAD16"
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

    if element_name == "Nd2":
        shapef["vol"] = Nd2()
        shapef["area"] = None
    elif element_name == "Nd3":
        shapef["vol"] = Nd3()
        shapef["area"] = None
    elif element_name == "Nd4":
        shapef["vol"] = Nd4()
        shapef["area"] = None
    elif element_name == "TRI3":
        shapef["vol"] = Tri3()
        shapef["area"] = Nd2()
    elif element_name == "TRI6":
        shapef["vol"] = Tri6()
        shapef["area"] = Nd3()
    elif element_name == "TRI10":
        shapef["vol"] = Tri10()
        shapef["area"] = Nd4()
    elif element_name == "QUAD4":
        shapef["vol"] = Quad4()
        shapef["area"] = Nd2()
    elif element_name == "QUAD8":
        shapef["vol"] = Quad8()
        shapef["area"] = Nd3()
    elif element_name == "QUAD9":
        shapef["vol"] = Quad9()
        shapef["area"] = Nd3()
    elif element_name == "QUAD12":
        shapef["vol"] = Quad12()
        shapef["area"] = Nd4()
    elif element_name == "QUAD16":
        shapef["vol"] = Quad16()
        shapef["area"] = Nd4()
    elif element_name == "TET4":
        shapef["vol"] = Tet4()
        shapef["area"] = Tri3()
    elif element_name == "HEXA8":
        shapef["vol"] = Hexa8()
        shapef["area"] = Quad4()
    else:
        logger.error("Invalid element name: {}".format(element_name))
        sys.exit(1)

    return shapef
