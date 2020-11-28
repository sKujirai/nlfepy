from .get_material_list import get_material_list, get_material


def Material(mater) -> list:
    """
    Get material classes

    Parameters
    ----------
    mater : list or string
        Material names

    Returns
    -------
    mater : list
        List of material class
    """

    if isinstance(mater, str):
        return [get_material(mater)]
    else:
        return get_material_list(mater)
