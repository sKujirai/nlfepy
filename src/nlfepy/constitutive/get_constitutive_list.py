import sys
from logging import getLogger
from .isotropic import Isotropic
from ..material import get_material


def get_constitutive_list(cnst_dict: dict, val: dict = {}) -> list:
    """
    Get material classes

    Parameters
    ----------
    cnst_dict : dict
        ('material', 'constitutive equation')

    Returns
    -------
    cnst : list
        List of Constitutive equation class
    """

    cnsts = []

    for mater, citems in cnst_dict.items():
        cname = citems[0]
        cparams = citems[1] if len(citems) > 1 else {}
        if cname == 'isotropic':
            cnsts.append(Isotropic(get_material(mater), val, cparams))
        else:
            logger = getLogger('Constitutive')
            logger.error('Invalid constitutive type: {}'.format(cnsts))
            sys.exit(1)

    return cnsts
