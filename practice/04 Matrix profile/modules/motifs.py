import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE
    
    mp = matrix_profile['mp']
    mp_len = len(mp)

    mp_temp = mp.copy()

    for i in range(top_k):
        min_idx = np.argmin(mp_temp)
        min_dist = mp_temp[min_idx]

        if 'pi' in matrix_profile:
            nn_idx = matrix_profile['pi'][min_idx]
        else:
            nn_idx = min_idx
        
        motif_pair = sorted([min_idx, nn_idx])
        motifs_idx.append(motif_pair)
        motifs_dist.append(min_dist)

        excl_zone = matrix_profile.get('excl_zone', mp_len // 2)
        exclusion_value = np.max(mp_temp)

        mp_temp = apply_exclusion_zone(mp_temp, motif_pair[0], excl_zone, exclusion_value)
        mp_temp = apply_exclusion_zone(mp_temp, motif_pair[1], excl_zone, exclusion_value)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
