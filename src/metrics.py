"""Clustering validity indices evaluated on a shared reference graph.

AVI/AVU: Shalileh et al., DOI 10.1134/S1064562425700589, eq. 18-21.
Apply the published adjacency sums to nonnegative symmetric weights.
S_Dbw: Halkidi-Vazirgiannis density-within-radius formulation; population
variance and point counts within the radius (see methodology notes).
MQ: weighted Newman-Girvan modularity, gamma=1 for comparable evaluation.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def graph_indices(adjacency, labels):
    labels = np.unique(labels, return_inverse=True)[1]
    k = int(labels.max()) + 1
    membership = np.eye(k)[labels]
    block = membership.T @ (adjacency @ membership)
    internal = np.diag(block)
    volume = block.sum(axis=1)
    external = volume - internal
    isolability = np.divide(internal, volume, out=np.zeros(k), where=volume > 0)
    denominator = external[:, None] + external[None, :] - block
    unifiability = np.divide(block, denominator, out=np.zeros_like(block), where=denominator > 1e-12)
    np.fill_diagonal(unifiability, 0)
    total = volume.sum()
    mq = float((internal / total - (volume / total) ** 2).sum()) if total else float('nan')
    return {"AVI": float(isolability.mean()), "AVU": float(unifiability.sum() / k), "MQ": mq}


def s_dbw(x, labels):
    groups = [x[labels == c] for c in np.unique(labels)]
    k = len(groups)
    variance = np.array([np.linalg.norm(g.var(axis=0)) for g in groups])
    norm_all = np.linalg.norm(x.var(axis=0))
    if k < 2 or k >= len(x) or norm_all <= 0:
        return float('nan')
    radius = float(np.sqrt(variance.sum()) / k)
    centers = np.array([g.mean(axis=0) for g in groups])
    density = [int(np.sum(np.linalg.norm(g - c, axis=1) <= radius)) for g, c in zip(groups, centers)]
    between = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            union = np.concatenate([groups[i], groups[j]])
            count = int(np.sum(np.linalg.norm(union - (centers[i] + centers[j]) / 2, axis=1) <= radius))
            denom = max(density[i], density[j])
            if denom == 0 and count:
                return float('inf')
            between += count / denom if denom else 0.0
    return float(variance.mean() / norm_all + 2 * between / (k * (k - 1)))


def validity(x, labels, reference, distances=None):
    labels = np.asarray(labels)
    sizes = np.unique(labels, return_counts=True)[1]
    out = {"clusters": len(sizes), "minimum_size": int(sizes.min()), "maximum_size": int(sizes.max())}
    if not 1 < len(sizes) < len(x):
        return {**out, **{m: float('nan') for m in ['SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'MQ']}}
    out['SW'] = float(silhouette_score(distances, labels, metric='precomputed') if distances is not None else silhouette_score(x, labels))
    out['CH'] = float(calinski_harabasz_score(x, labels))
    out['CH_per_N'] = out['CH'] / len(x)
    out['S_Dbw'] = s_dbw(x, labels)
    out.update(graph_indices(reference, labels))
    return out
