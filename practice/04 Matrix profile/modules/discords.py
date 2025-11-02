import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # INSERT YOUR CODE

    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi'].copy()
    
    excl_zone = matrix_profile.get('ez', 0)
    if excl_zone == 0:
        sublen = matrix_profile.get('w', matrix_profile.get('sublen', matrix_profile.get('sublen', 1)))
        excl_zone = int(sublen/2)
    
    for _ in range(top_k):
        max_idx = np.argmax(mp)
        max_val = mp[max_idx]

        if max_val == -np.inf or max_val == np.inf or np.isnan(max_val):
            break
        
        discords_idx.append(max_idx)
        discords_dist.append(max_val)
        discords_nn_idx.append(mpi[max_idx])

        mp = apply_exclusion_zone(mp, max_idx, excl_zone, -np.inf)
        
    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
