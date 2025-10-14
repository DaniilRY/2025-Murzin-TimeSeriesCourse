import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    if len(ts1) != len(ts2):
        raise ValueError("Длина временных рядов не совпадает")

    # Вычисляем евклидово расстояние
    diff = ts1 - ts2
    squared_diff = np.sum(diff ** 2)
    ed_dist = np.sqrt(squared_diff)

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n, m = len(ts1), len(ts2)
    w = int(max(r * max(n, m), abs(n - m)))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        start = max(1, i - w)
        end = min(m, i + w) + 1
        
        for j in range(start, end):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j  ],
                                   dtw[i  , j-1],
                                   dtw[i-1, j-1])

    return dtw[n, m] 
