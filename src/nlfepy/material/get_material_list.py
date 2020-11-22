import sys
from logging import getLogger
from .aluminium import Aluminium


def get_material_list(material_names: list) -> list:
    """
    Get material classes

    Parameters
    ----------
    material names : list
        Material names

    Returns
    -------
    mater : list
        List of material class
    """

    maters = []

    for mname in material_names:
        if mname == 'Al' or mname == 'Aluminium':
            maters.append(Aluminium())
        else:
            logger = getLogger('Material')
            logger.error('Invalid material: {}'.format(mname))
            sys.exit(1)

    return maters
