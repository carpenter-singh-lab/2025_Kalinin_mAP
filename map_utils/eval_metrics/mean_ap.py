"""Calculate mean average precision (mAP).

Based on the implementation in copairs:
https://github.com/cytomining/copairs/blob/main/src/copairs/map/map.py
"""

from itertools import combinations, product

import numpy as np
import numpy.typing as npt

from copairs import compute
from copairs.map.average_precision import build_rank_lists


def copairs_similarity(
    pos_pairs: npt.ArrayLike,
    neg_pairs: npt.ArrayLike,
    feats: npt.ArrayLike,
    batch_size: int = 20000,
    distance: str = "cosine",
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Compute similarity for positive and negative pairs."""
    pairwise_distance = compute.get_similarity_fn(distance)
    pos_sims = pairwise_distance(feats, pos_pairs, batch_size)
    neg_sims = pairwise_distance(feats, neg_pairs, batch_size)
    return pos_sims, neg_sims


def get_p_value(
    map_score: float,
    indices: npt.ArrayLike,
    null_dists: npt.ArrayLike,
    null_size: int,
    rev_ix: npt.ArrayLike,
) -> float:
    """Calculate p-value for mAP score."""
    null_dist = null_dists[rev_ix[indices]].mean(axis=0)
    num = (null_dist > map_score).sum()
    p_value = (num + 1) / (null_size + 1)
    return p_value


def mean_ap(
    pert_features: npt.ArrayLike,
    control_features: npt.ArrayLike,
    distance="cosine",
    null_size: int = 1000,
    random_seed: int = 0,
) -> float:
    """Calculate mean average precision (mAP) for a given group."""
    feats = np.vstack([pert_features, control_features])
    pos_pairs = np.array(list(combinations(range(len(pert_features)), 2)))
    neg_pairs = np.array(
        list(
            product(
                range(len(pert_features)),
                np.arange(len(control_features)) + len(pert_features),
            )
        )
    )
    pos_sims, neg_sims = copairs_similarity(
        pos_pairs, neg_pairs, feats, distance=distance
    )

    _, rel_k_list, counts = build_rank_lists(pos_pairs, neg_pairs, pos_sims, neg_sims)
    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
    null_confs_unique, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = compute.get_null_dists(null_confs_unique, null_size, seed=random_seed)

    map_score = ap_scores[: len(pert_features)].mean()
    p_value = get_p_value(
        map_score, range(len(pert_features)), null_dists, null_size, rev_ix
    )
    return p_value
