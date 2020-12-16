import sys
from logging import getLogger
from .aluminium import Aluminium


def get_material(material_name: str) -> list:
    """
    Get material class

    Parameters
    ----------
    material name : str
        Material name

    Returns
    -------
    mater :
        Material class
    """

    if material_name == "Al" or material_name == "Aluminium":
        return Aluminium()
    else:
        logger = getLogger("Material")
        logger.error("Invalid material: {}".format(material_name))
        sys.exit(1)


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
        maters.append(get_material(mname))

    return maters
