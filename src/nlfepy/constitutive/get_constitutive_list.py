import sys
from logging import getLogger
from .isotropic import Isotropic
from .j2flow import J2flow
from .crystal_plasticity import CrystalPlasticity
from ..material import get_material


def get_constitutive_list(cnst_dict: dict, *, nitg: int, val: dict = {}) -> list:
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

    cnsts = []

    for mater, citems in cnst_dict.items():
        cname = citems[0]
        cparams = citems[1] if len(citems) > 1 else {}
        if cname == 'isotropic':
            cnsts.append(Isotropic(metal=get_material(mater), nitg=nitg, val=val, params=cparams))
        elif cname == 'j2flow':
            cnsts.append(J2flow(metal=get_material(mater), nitg=nitg, val=val, params=cparams))
        elif cname == 'crystal_plasticity':
            cnsts.append(CrystalPlasticity(metal=get_material(mater), nitg=nitg, val=val, params=cparams))
        else:
            logger = getLogger('Constitutive')
            logger.error('Invalid constitutive type: {}'.format(cnsts))
            sys.exit(1)

    return cnsts
