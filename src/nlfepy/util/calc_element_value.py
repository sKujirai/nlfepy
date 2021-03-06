import numpy as np


def calc_element_value(*, mesh, values: dict, keys: list = []):
    """
    Calc. physical quantities of each element

    Parameters
    ----------
    mesh :
        Mesh class (See mesh.py)
    values : dict
        Physical quantities {'point', 'elemnt', 'itg'}
    keys : list
        Values to calc. element value (if list is empty, calc. all values)
    """

    elm_keys = []
    for key in keys:
        if key in values["itg"].keys():
            elm_keys.append(key)
    if len(elm_keys) == 0:
        elm_keys = values["itg"].keys()

    for key in elm_keys:
        if values["itg"][key].ndim == 1:
            values["element"][key] = np.zeros(mesh.n_element)
        else:
            values["element"][key] = np.zeros(
                (mesh.n_element, values["itg"][key][0].flatten().shape[0])
            )

    for ielm in range(mesh.n_element):
        n_intgp = mesh.n_intgp("vol", elm=ielm)

        itg = mesh.itg_idx(elm=ielm)
        weight = mesh.get_wdetJ("vol", elm=ielm)
        for key in elm_keys:
            values["element"][key][ielm] = np.dot(
                values["itg"][key][itg].reshape(n_intgp, -1).T, weight
            )
        for key in elm_keys:
            values["element"][key][ielm] /= np.sum(weight)
