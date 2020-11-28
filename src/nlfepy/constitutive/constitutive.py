from .get_constitutive_list import get_constitutive_list


def Constitutive(cnst_dict: dict, *, nitg: int, val: dict = {}) -> list:
    """
    Get material classes

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
