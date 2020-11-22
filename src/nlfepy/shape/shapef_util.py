from ..shape import Nd2, Tri3, Tri6, Quad4, Hexa8
import sys
from logging import getLogger


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

    logger = getLogger('shape')

    element_name = None

    if n_dof == 2:
        if n_node == 3:
            element_name = 'Tri3'
        elif n_node == 4:
            element_name = 'Quad4'
        elif n_node == 6:
            element_name = 'Tri6'
        else:
            logger.error('Invalid 2D element')
            sys.exit(1)
    else:
        if n_node == 8:
            element_name = 'Hexa8'
        else:
            logger.error('Invalid 2D element')
            sys.exit(1)

    return element_name


def get_shape_function(element_name: str) -> dict:
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

    logger = getLogger('shape')

    shapef = {}

    if element_name == 'Tri3':
        shapef['vol'] = Tri3()
        shapef['area'] = Nd2()
    elif element_name == 'Tri6':
        shapef['vol'] = Tri6()
        shapef['area'] = Nd2()
    elif element_name == 'Quad4':
        shapef['vol'] = Quad4()
        shapef['area'] = Nd2()
    elif element_name == 'Hexa8':
        shapef['vol'] = Hexa8()
        shapef['area'] = Quad4()
    else:
        logger.error('Invalid element name: {}'.format(element_name))
        sys.exit(1)

    return shapef
