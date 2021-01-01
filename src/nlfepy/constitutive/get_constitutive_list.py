import sys
from logging import getLogger
from functools import singledispatch
from typing import List
from .constitutive_base import ConstitutiveBase
from .isotropic import Isotropic
from .j2flow import J2flow
from .crystal_plasticity import CrystalPlasticity
from ..material import get_material


@singledispatch
def get_constitutive_list(
    cnst_dict: dict, *, nitg: int, val: dict = {}
) -> List[ConstitutiveBase]:
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

    cnsts: List[ConstitutiveBase] = []

    for mater, citems in cnst_dict.items():
        cname = citems[0]
        cparams = citems[1] if len(citems) > 1 else {}
        if cname == "isotropic":
            cnsts.append(
                Isotropic(metal=get_material(mater), nitg=nitg, val=val, params=cparams)
            )
        elif cname == "j2flow":
            cnsts.append(
                J2flow(metal=get_material(mater), nitg=nitg, val=val, params=cparams)
            )
        elif cname == "crystal_plasticity":
            cnsts.append(
                CrystalPlasticity(
                    metal=get_material(mater), nitg=nitg, val=val, params=cparams
                )
            )
        else:
            logger = getLogger("Constitutive")
            logger.error("Invalid constitutive type: {}".format(cnsts))
            sys.exit(1)

    return cnsts


@get_constitutive_list.register(list)
def _(maters: list, *, nitg: int, val: dict = {}) -> List[ConstitutiveBase]:
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

    cnsts: List[ConstitutiveBase] = []

    for mater in maters:
        cnsts.append(Isotropic(metal=mater, nitg=nitg, val=val))

    return cnsts
