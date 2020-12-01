from functools import singledispatch
from .get_constitutive_list import get_constitutive_list


@singledispatch
def Constitutive(cnst_dict: dict, *, nitg: int, val: dict = {}) -> list:
    """
    Get constitutive classes

    Parameters
    ----------
    cnst_dict : dict
        ('material', 'constitutive equation')
    ntig : int
        Total number of integral points
    val : dict
        Physical quantities

    Returns
    -------
    cnst : list
        List of Constitutive equation class
    """

    return get_constitutive_list(cnst_dict, nitg=nitg, val=val)


@Constitutive.register(list)
def _(maters: list, *, nitg: int, val: dict = {}) -> list:
    """
    Get constitutive classes

    Parameters
    ----------
    maters : list
        List of material class (See material.py)
    ntig : int
        Total number of integral points
    val : dict
        Physical quantities

    Returns
    -------
    cnst : list
        List of Constitutive equation class
    """

    return get_constitutive_list(maters, nitg=nitg, val=val)
