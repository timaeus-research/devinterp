import numpy as np


def ith_place_first_diff(ary, i):
    return -np.subtract(ary[:, :-i, :], ary[:, i:, :])


# Marginal distribution of n-th jet coordinate
def ith_place_nth_diff(ary, i, n=1):
    n_th_diff = ary
    for _ in range(n):
        n_th_diff = ith_place_first_diff(n_th_diff, i)
    return n_th_diff


# Joint distribution of n-jets
def joint_ith_place_nth_diff(ary, i, n=1):
    tot_length = ary.shape[1]
    list_jet_coordinates = [ary[:, : tot_length - n * i, :]]
    m_th_diff = ary
    for m in range(n):
        m_th_diff = ith_place_first_diff(m_th_diff, i)
        list_jet_coordinates.append(m_th_diff[:, : tot_length - n * i, :])
    return np.concatenate(list_jet_coordinates, axis=2)
