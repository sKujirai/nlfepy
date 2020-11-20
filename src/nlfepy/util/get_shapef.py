from ..shape import Nd2, Tri3, Tri6, Quad4, Hexa8
import sys
from logging import getLogger


def get_shape_function(*, n_dof: int, n_node: int) -> dict:
    """
    Get shape function

    Parameters
    ----------
    n_dof : int
        Number of degrees of freedom
    n_node : int
        Number of nodes of each element

    Returns
    -------
    shapef : dict
        Dictionary of shape function
    """

    logger = getLogger('shape')

    shapef = {}

    if n_dof == 2:
        if n_node == 3:
            shapef['vol'] = Tri3()
        elif n_node == 4:
            shapef['vol'] = Quad4()
        elif n_node == 6:
            shapef['vol'] = Tri6()
        else:
            logger.error('Invalid 2D element')
            sys.exit(1)
        shapef['area'] = Nd2()
    else:
        if n_node == 8:
            shapef['vol'] = Hexa8()
        else:
            logger.error('Invalid 2D element')
            sys.exit(1)
        shapef['area'] = Quad4()

    return shapef
