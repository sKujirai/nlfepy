import sys
from logging import getLogger
from typing import List
from .material_base import MaterialBase
from .aluminium import Aluminium


def get_material(material_name: str) -> MaterialBase:
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


def get_material_list(material_names: List[str]) -> List[MaterialBase]:
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

    maters: List[MaterialBase] = []

    for mname in material_names:
        maters.append(get_material(mname))

    return maters
