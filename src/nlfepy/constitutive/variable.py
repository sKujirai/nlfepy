from typing import Dict


def Variable() -> Dict[str, dict]:
    """
    Generate dictionary of physical quantities

    Returns
    -------
    vals : Dict[str, dict]
        Dictionary of variables ['point', 'element', 'itg']
    """

    vals: Dict[str, dict] = {
        # Values in global nodes
        "point": {},
        # Values in each element
        "element": {},
        # Values in each integral points
        "itg": {},
    }

    return vals
