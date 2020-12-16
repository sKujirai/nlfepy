def Variable() -> dict:
    """
    Generate dictionary of physical quantities

    Returns
    -------
    vals : dict
        Dictionary of variables ['point', 'element', 'itg']
    """

    vals = {
        # Values in global nodes
        "point": {},
        # Values in each element
        "element": {},
        # Values in each integral points
        "itg": {},
    }

    return vals
