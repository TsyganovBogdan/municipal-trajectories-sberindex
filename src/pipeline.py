"""Municipal trajectories: a reproducible dynamic attributed-network study.

Run from the project root: python -m src.pipeline --config configs/default.json
All models are selected on development snapshots only. Confirmation months
are evaluated after selection; the independent market-access indicator is not
an input to model selection or feature construction.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from importlib.metadata import version

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from .metrics import validity, graph_indices

CATEGORIES = ['Продовольствие', 'Здоровье', 'Маркетплейсы', 'Общественное питание', 'Транспорт']
ALL = 'Все категории'
SHARES = CATEGORIES + ['Прочие расходы']
METRICS = ['SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'MQ']


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_data(raw, out):
    c = pd.read_parquet(raw / 'consumption.parquet')
    if 'consumption' in c and 'value' not in c:
        c = c.rename(columns={'consumption': 'value'})
    assert not c.duplicated(['territory_id', 'date', 'category']).any(), 'Duplicate consumption keys'
    assert c['value'].notna().all() and (c['value'] > 0).all(), 'Invalid consumption values'
    p = c.pivot(index=['territory_id', 'date'], columns='category', values='value').sort_index()
    months = sorted(c['date'].unique())
    if months != list(pd.period_range('2023-01', '2024-12', freq='M').astype(str)):
        raise ValueError('This frozen study expects January 2023 - December 2024; revise config for other data.')
    complete = p.groupby(level=0).apply(lambda x: len(x) == len(months) and x.notna().all().all())
    ids = complete[complete].index.to_numpy(dtype=int)
    audit_territories = p.groupby(level=0).size().rename('months_available').reset_index()
    audit_territories['included'] = audit_territories.territory_id.isin(ids)
    audit_territories.to_csv(out / 'territory_coverage.csv', index=False)
    p.groupby(level=1).size().rename('territories').to_csv(out / 'monthly_coverage.csv')
    levels = np.stack([p.xs(m, level=1).loc[ids, [ALL] + CATEGORIES].to_numpy(float) for m in months])
    remainder = levels[:, :, 0] - levels[:, :, 1:].sum(axis=2)
    assert (remainder >= 0).all(), 'Category overlap or inconsistent total'
    parts = np.concatenate([levels[:, :, 1:], remainder[:, :, None]], axis=2)
    shares = parts / levels[:, :, :1]
    np.testing.assert_allclose(shares.sum(axis=2), 1)
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(raw.glob('*')) if f.is_file()}
    audit = {
        'consumption_rows': len(c), 'territories_in_raw': int(c.territory_id.nunique()),
        'balanced_territories': len(ids), 'excluded_territories': int((~complete).sum()),
        'months': months, 'missing_fields': int(c.isna().sum().sum()), 'duplicate_keys': 0,
        'minimum_remainder_rub': float(remainder.min()), 'balanced_fraction': float(len(ids) / c.territory_id.nunique()),
        'value_field_note': 'Файл использует value; описание называет поле consumption. Адаптер обрабатывает оба имени.',
        'boundary_note': 'По описанию СберИндекса territory_id относится к постоянным границам. Независимая проверка геометрии не выполнена.',
        'hashes': manifest,
    }
    save_json(out / 'data_audit.json', audit)
    return ids, months, levels, shares, audit


def build_features(levels, shares, months, development):
    # Three-month trailing averages end at the current observation; no future smoothing.
    smooth = np.stack([shares[max(0, t - 2):t + 1].mean(axis=0) for t in range(24)])
    total_log = np.log(levels[:, :, 0])
    relative_level = total_log - np.median(total_log, axis=1, keepdims=True)
    profile = np.concatenate([np.sqrt(smooth[12:]), relative_level[12:, :, None]], axis=2)
    # Log year-on-year changes for six expenditure indicators; subtract the
    # cross-sectional median to remove their common national nominal movement.
    growth = np.log(levels[12:] / levels[:12])
    growth = growth - np.median(growth, axis=1, keepdims=True)
    dev_idx = [months.index(m) - 12 for m in development]
    sp = RobustScaler().fit(np.concatenate([profile[t] for t in dev_idx]))
    sg = RobustScaler().fit(np.concatenate([growth[t] for t in dev_idx]))
    pp = np.stack([np.clip(sp.transform(v), -6, 6) for v in profile]) / np.sqrt(profile.shape[2])
    gg = np.stack([np.clip(sg.transform(v), -6, 6) for v in growth]) / np.sqrt(growth.shape[2])
    xx = np.concatenate([pp / np.sqrt(2), gg / np.sqrt(2)], axis=2)
    meta = {
        'profile_columns': ['sqrt_share_' + c for c in SHARES] + ['relative_log_total'],
        'growth_columns': ['relative_log_yoy_' + c for c in [ALL] + CATEGORIES],
        'profile_median': sp.center_.tolist(), 'profile_iqr': sp.scale_.tolist(),
        'growth_median': sg.center_.tolist(), 'growth_iqr': sg.scale_.tolist(),
        'fit_months': development, 'clipping': [-6, 6],
        'view_weight': {'profile': 0.5, 'growth': 0.5},
        'note': 'Cross-sectional peer comparison is retrospective and not a forecast; values are nominal.'
    }
    return pp, gg, xx, meta


def knn_graph(distances, neighbors):
    d = np.array(distances, copy=True)
    np.fill_diagonal(d, np.inf)
    n = len(d)
    k = min(neighbors, n - 1)
    inds = np.argpartition(d, k - 1, axis=1)[:, :k]
    row = np.repeat(np.arange(n), k)
    col = inds.ravel()
    local = d[row, col]
    tau = np.median(local[np.isfinite(local) & (local > 0)])
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0
    weights = np.exp(-local / tau)
    weights[~np.isfinite(local)] = 0
    a = sparse.csr_matrix((weights, (row, col)), shape=(n, n))
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def leiden(a, resolution, seed):
    edges = sparse.triu(a, 1).tocoo()
    g = ig.Graph(n=a.shape[0], edges=list(zip(edges.row.tolist(), edges.col.tolist())), directed=False)
    result = la.find_partition(g, la.RBConfigurationVertexPartition, weights=edges.data.tolist(),
                               resolution_parameter=resolution, n_iterations=-1, seed=int(seed))
    return np.asarray(result.membership, dtype=int)


def align_labels(previous, current):
    old = np.unique(previous)
    new = np.unique(current)
    overlap = np.array([[np.sum((previous == i) & (current == j)) for j in new] for i in old])
    r, c = linear_sum_assignment(-overlap)
    mapping = {new[j]: int(old[i]) for i, j in zip(r, c)}
    nxt = int(old.max()) + 1
    for j in new:
        if j not in mapping:
            mapping[j] = nxt
            nxt += 1
    return np.array([mapping[v] for v in current])


def candidate_list(cfg):
    result = []
    for view in cfg['graph_views']:
        for k in cfg['neighbors']:
            for resolution in cfg['resolutions']:
                result.append({'id': f'Leiden_{view}_k{k}_r{resolution}', 'method': 'Leiden', 'view': view, 'neighbors': k, 'resolution': resolution})
    for method in ['KMeans', 'Ward']:
        for k in cfg['baseline_clusters']:
            result.append({'id': f'{method}_{k}', 'method': method, 'clusters': k})
    return result


def partition(spec, x, graphs, seed):
    if spec['method'] == 'Leiden':
        return leiden(graphs[(spec['view'], spec['neighbors'])], spec['resolution'], seed)
    if spec['method'] == 'KMeans':
        return KMeans(n_clusters=spec['clusters'], n_init=15, random_state=seed).fit_predict(x)
    return AgglomerativeClustering(n_clusters=spec['clusters'], linkage='ward').fit_predict(x)


def distance_views(p, g, x):
    return {'profile': cdist(p, p), 'growth': cdist(g, g), 'combined': cdist(x, x)}


def get_graphs(ds, cfg):
    return {(view, k): knn_graph(ds[view], k) for view in cfg['graph_views'] for k in cfg['neighbors']}


def select_model(pp, gg, xx, months24, cfg, out):
    candidates = candidate_list(cfg)
    rows, stability = [], []
    for month in cfg['development_months']:
        t = months24.index(month)
        ds = distance_views(pp[t], gg[t], xx[t])
        graphs = get_graphs(ds, cfg)
        ref = knn_graph(ds['combined'], cfg['reference_neighbors'])
        for spec in candidates:
            started = time.perf_counter()
            labels = partition(spec, xx[t], graphs, cfg['seed'])
            met = validity(xx[t], labels, ref, ds['combined'])
            rows.append({'candidate': spec['id'], 'month': month, **met, 'seconds': time.perf_counter() - started})
            if month == cfg['development_months'][-1]:
                for seed in cfg['stability_seeds'][:3]:
                    lab2 = partition(spec, xx[t], graphs, seed)
                    stability.append({'candidate': spec['id'], 'seed': seed, 'ARI': adjusted_rand_score(labels, lab2)})
        print(f'Development {month}: {len(candidates)} candidates complete', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'model_comparison_development.csv', index=False)
    st = pd.DataFrame(stability)
    st.to_csv(out / 'selection_seed_stability.csv', index=False)
    summary = frame.groupby('candidate')[METRICS + ['clusters', 'minimum_size', 'seconds']].mean()
    summary['seed_ARI'] = st.groupby('candidate').ARI.mean()
    ranges = frame.groupby('candidate').agg(min_K=('clusters', 'min'), max_K=('clusters', 'max'), min_group=('minimum_size', 'min'))
    summary = summary.join(ranges)
    lo, hi = cfg['allowed_cluster_count']
    summary['eligible'] = (summary.min_K >= lo) & (summary.max_K <= hi) & (summary.min_group >= cfg['minimum_cluster_size'])
    # Predeclared balanced percentile score. Individual metrics stay visible.
    # 40% attribute validity, 40% common-reference graph validity, 20% seed stability.
    ranked = []
    for month, part in frame.groupby('month'):
        part = part.set_index('candidate')
        ranks = pd.DataFrame({m: part[m].rank(pct=True, ascending=m not in ['S_Dbw', 'AVU']) for m in METRICS})
        ranked.append(0.4 * ranks[['SW', 'CH', 'S_Dbw']].mean(axis=1) + 0.4 * ranks[['AVI', 'AVU', 'MQ']].mean(axis=1))
    summary['selection_score'] = pd.concat(ranked, axis=1).mean(axis=1) + 0.2 * summary.seed_ARI.rank(pct=True)
    summary = summary.sort_values('selection_score', ascending=False)
    summary.to_csv(out / 'selection_summary.csv')
    eligible = summary[summary.eligible]
    if eligible.empty:
        raise RuntimeError('No model satisfies the configured cluster-count and minimum-size constraints.')
    best_id = eligible.index[0]
    best = next(x for x in candidates if x['id'] == best_id)
    # A graph-based comparison is essential even if an attribute baseline wins.
    graph_best_id = eligible[eligible.index.str.startswith('Leiden')].index[0]
    graph_best = next(x for x in candidates if x['id'] == graph_best_id)
    selected = {'best_overall': best, 'graph_model': graph_best, 'score_rule': '0.4 attribute percentiles + 0.4 reference-graph percentiles + 0.2 seed-ARI percentile; development only'}
    save_json(out / 'selection.json', selected)
    return best, graph_best, candidates, summary


def road_data(raw, ids, out):
    connections = pd.read_parquet(raw / 'connection.parquet')
    audit = []
    for typ, d in connections.groupby('type'):
        low = np.minimum(d.territory_id_x.to_numpy(), d.territory_id_y.to_numpy())
        high = np.maximum(d.territory_id_x.to_numpy(), d.territory_id_y.to_numpy())
        unordered = pd.MultiIndex.from_arrays([low, high])
        audit.append({'type': typ, 'rows': len(d), 'territories': len(set(low) | set(high)),
                      'duplicate_undirected_pairs': int(unordered.duplicated().sum()), 'zero_distance': int((d.distance == 0).sum())})
    pd.DataFrame(audit).to_csv(out / 'connection_audit.csv', index=False)
    road = connections[connections.type == 'highway'].copy()
    road['lo'] = np.minimum(road.territory_id_x, road.territory_id_y)
    road['hi'] = np.maximum(road.territory_id_x, road.territory_id_y)
    road = road.groupby(['lo', 'hi'], as_index=False).distance.mean()
    idx = {int(v): i for i, v in enumerate(ids)}
    road = road[road.lo.isin(idx) & road.hi.isin(idx)]
    r = road.lo.map(idx).to_numpy(int)
    c = road.hi.map(idx).to_numpy(int)
    d = np.full((len(ids), len(ids)), np.inf, dtype=np.float32)
    d[r, c] = road.distance.to_numpy()
    d[c, r] = road.distance.to_numpy()
    np.fill_diagonal(d, 0)
    return d


def main(config_path):
    start = time.perf_counter()
    cfg = json.loads(Path(config_path).read_text())
    raw, out = Path(cfg['input_dir']), Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    ids, months, levels, shares, audit = load_data(raw, out)
    pp, gg, xx, transformations = build_features(levels, shares, months, cfg['development_months'])
    save_json(out / 'transformations.json', transformations)
    months24 = months[12:]
    best, graph_best, candidates, summary = select_model(pp, gg, xx, months24, cfg, out)
    print('Selected overall:', best, 'selected graph:', graph_best, flush=True)
    # Use the selected graph model for network analysis; retain the overall
    # winner and two attribute-based alternatives for comparison.
    primary = graph_best
    alternatives = [next(s for s in candidates if s['id'] == name) for name in summary.index if name.startswith(('KMeans', 'Ward'))][:2]
    compare_specs = [primary] + [s for s in [best] + alternatives if s['id'] != primary['id']]
    compare_specs = list({s['id']: s for s in compare_specs}.values())
    all_labels, memberships, quality, stability_rows, transitions, peer_rows, growth_rows = [], [], [], [], [], [], []
    monthly_graphs, support_list, distances_list = [], [], []
    rng = np.random.default_rng(cfg['seed'])
    for t, month in enumerate(months24):
        ds = distance_views(pp[t], gg[t], xx[t])
        graphs = get_graphs(ds, cfg)
        ref = knn_graph(ds['combined'], cfg['reference_neighbors'])
        labels = partition(primary, xx[t], graphs, cfg['seed'])
        labels = align_labels(all_labels[-1], labels) if all_labels else labels
        all_labels.append(labels)
        monthly_graphs.append(graphs[(primary['view'], primary['neighbors'])])
        distances_list.append(ds['combined'])
        partition_samples = []
        # Assess sensitivity to seeds, neighborhood size and resolution.
        # Align variants to this month's primary partition.
        for seed in cfg['stability_seeds']:
            v = dict(primary)
            y = partition(v, xx[t], graphs, seed)
            stability_rows.append({'month': month, 'variant': f'seed_{seed}', 'ARI': adjusted_rand_score(labels, y)})
            partition_samples.append(align_labels(labels, y))
        for delta in [-10, 10]:
            k = max(5, primary['neighbors'] + delta)
            a = knn_graph(ds[primary['view']], k)
            y = leiden(a, primary['resolution'], cfg['seed'])
            stability_rows.append({'month': month, 'variant': f'neighbors_{k}', 'ARI': adjusted_rand_score(labels, y)})
            partition_samples.append(align_labels(labels, y))
        for factor in [0.9, 1.1]:
            y = leiden(graphs[(primary['view'], primary['neighbors'])], primary['resolution'] * factor, cfg['seed'])
            stability_rows.append({'month': month, 'variant': f'resolution_x{factor}', 'ARI': adjusted_rand_score(labels, y)})
            partition_samples.append(align_labels(labels, y))
        support = np.mean(np.stack(partition_samples) == labels, axis=0)
        support_list.append(support)
        sil = silhouette_samples(ds['combined'], labels, metric='precomputed')
        stage = 'development' if month in cfg['development_months'] else 'validation' if month in cfg['validation_months'] else 'confirmation' if month in cfg['confirmation_months'] else 'descriptive'
        for spec in compare_specs:
            y = labels if spec['id'] == primary['id'] else partition(spec, xx[t], graphs, cfg['seed'])
            quality.append({'month': month, 'stage': stage, 'candidate': spec['id'], **validity(xx[t], y, ref, ds['combined'])})
        for i, territory in enumerate(ids):
            memberships.append({'territory_id': int(territory), 'month': month, 'cluster': int(labels[i]),
                                'support': float(support[i]), 'silhouette': float(sil[i]),
                                'total_rub': float(levels[t + 12, i, 0]),
                                'total_yoy_pct': float(100 * (levels[t + 12, i, 0] / levels[t, i, 0] - 1)),
                                **{f'share_{c}': float(shares[t + 12, i, j]) for j, c in enumerate(SHARES)}})
        print(f'Dynamic analysis {month}: K={len(np.unique(labels))}, seed/parameter variants=9', flush=True)
    labels_array = np.stack(all_labels)
    support_array = np.stack(support_list)
    membership = pd.DataFrame(memberships)
    membership.to_csv(out / 'memberships.csv', index=False)
    pd.DataFrame(quality).to_csv(out / 'monthly_quality.csv', index=False)
    pd.DataFrame(stability_rows).to_csv(out / 'stability.csv', index=False)
    # Full overlap table represents splits and merges, beyond one-to-one colors.
    for t in range(1, 12):
        for a in np.unique(labels_array[t - 1]):
            for b in np.unique(labels_array[t]):
                count = int(np.sum((labels_array[t - 1] == a) & (labels_array[t] == b)))
                if count:
                    transitions.append({'from_month': months24[t - 1], 'to_month': months24[t], 'from_cluster': int(a), 'to_cluster': int(b), 'count': count})
    pd.DataFrame(transitions).to_csv(out / 'cluster_flows.csv', index=False)
    events = []
    for t in range(1, 12):
        for i in np.flatnonzero(labels_array[t] != labels_array[t - 1]):
            # Two consecutive snapshots in the new cluster; December cannot
            # be confirmed because January 2025 is not present.
            persistent = bool(t < 11 and labels_array[t + 1, i] == labels_array[t, i])
            # Mean-standardized block-space displacement; not a p-value.
            effect = float(np.linalg.norm(xx[t, i] - xx[t - 1, i]))
            minimum_support = float(min(support_array[t - 1, i], support_array[t, i], support_array[min(t + 1, 11), i]))
            robust = persistent and minimum_support >= cfg['transition_support'] and effect >= cfg['transition_min_effect']
            events.append({'territory_id': int(ids[i]), 'month': months24[t], 'from_cluster': int(labels_array[t - 1, i]),
                           'to_cluster': int(labels_array[t, i]), 'persistent_two_months': persistent,
                           'support_min': minimum_support, 'feature_displacement': effect, 'robust': bool(robust),
                           'status': 'retrospectively_supported' if robust else 'unconfirmed' if t == 11 else 'sensitive_or_small',
                           'marketplace_share_change_pp': float(100 * (shares[t + 12, i, 2] - shares[t + 11, i, 2]))})
    ev = pd.DataFrame(events)
    ev.to_csv(out / 'transition_events.csv', index=False)
    road = road_data(raw, ids, out)
    # Road topology is dated 31 Dec 2024: use it as retrospective context, never
    # claim it was available before that date for real-time decisions.
    december = labels_array[-1]
    ref_dec = knn_graph(distances_list[-1], cfg['reference_neighbors'])
    geo_graph = knn_graph(road, cfg['reference_neighbors'])
    connected = np.asarray(geo_graph.sum(axis=1)).ravel() > 0
    geo_lab = leiden(geo_graph, primary['resolution'], cfg['seed'])
    geo_quality = validity(xx[-1], geo_lab, ref_dec, distances_list[-1])
    geo_quality['ARI_with_economic_partition'] = float(adjusted_rand_score(december[connected], geo_lab[connected]))
    geo_quality['road_connected_territories'] = int(connected.sum())
    save_json(out / 'geography_comparison.json', geo_quality)
    e = sparse.triu(monthly_graphs[-1], 1).tocoo()
    road_on_edges = road[e.row, e.col]
    finite = np.isfinite(road_on_edges)
    edge_summary = {'economic_edges_december': int(len(e.row)), 'edges_with_road_distance': int(finite.sum()),
                    'fraction_over_1000km': float(np.mean(road_on_edges[finite] > 1000)),
                    'median_road_km': float(np.median(road_on_edges[finite])),
                    'geographic_baseline': geo_quality}
    save_json(out / 'edge_distance_summary.json', edge_summary)
    order = np.argsort(-e.data)
    pd.DataFrame({'source': ids[e.row[order]], 'target': ids[e.col[order]], 'weight': e.data[order], 'road_km': road_on_edges[order]}).to_csv(out / 'economic_edges_december.csv', index=False)
    # Neighbors restricted to the selected cluster make interpretation clear;
    # distances are in the common standardized feature space.
    d = distances_list[-1].copy()
    np.fill_diagonal(d, np.inf)
    for i in range(len(ids)):
        same = np.flatnonzero((december == december[i]) & (np.arange(len(ids)) != i))
        peers = same[np.argsort(d[i, same])[:5]]
        for rank, j in enumerate(peers, 1):
            peer_rows.append({'territory_id': int(ids[i]), 'peer_id': int(ids[j]), 'rank': rank,
                              'feature_distance': float(d[i, j]), 'road_km': float(road[i, j]) if np.isfinite(road[i, j]) else None})
    pd.DataFrame(peer_rows).to_csv(out / 'peers_december.csv', index=False)
    # Context validation excluded from all feature/model-selection steps.
    market = pd.read_parquet(raw / 'market_access.parquet').set_index('territory_id').market_access.reindex(ids)
    ext = pd.DataFrame({'territory_id': ids, 'cluster': december, 'market_access': market.to_numpy()})
    ext.to_csv(out / 'external_validation.csv', index=False)
    finite_ma = market.notna().to_numpy()
    log_ma = np.log1p(market.to_numpy()[finite_ma])
    lab_ma = december[finite_ma]
    def eta2(values, labs):
        total = np.sum((values - values.mean()) ** 2)
        return float(sum(np.sum(labs == c) * (values[labs == c].mean() - values.mean()) ** 2 for c in np.unique(labs)) / total)
    eta = eta2(log_ma, lab_ma)
    # Descriptive permutation reference, not a spatially valid causal p-value.
    perm_eta = [eta2(log_ma, rng.permutation(lab_ma)) for _ in range(200)]
    rho = spearmanr(market.to_numpy()[finite_ma], shares[-1, finite_ma, 2]).statistic
    ext_summary = {'available': int(finite_ma.sum()), 'missing': int((~finite_ma).sum()),
                   'eta_squared_log_market_access': eta, 'permuted_eta_median': float(np.median(perm_eta)),
                   'market_access_vs_marketplace_share_spearman': float(rho),
                   'interpretation': 'Описательное сопоставление кластеров с индексом доступности рынков, исключённым из признаков модели. Перестановки сохраняют размеры групп без учёта пространственной зависимости; статистическая значимость по ним не оценивается.'}
    save_json(out / 'external_validation_summary.json', ext_summary)
    # Null labels preserve the chosen cluster size distribution exactly.
    random_rows = []
    for b in range(cfg['random_baselines']):
        random_rows.append({'repeat': b, **validity(xx[-1], rng.permutation(december), ref_dec, distances_list[-1])})
    pd.DataFrame(random_rows).to_csv(out / 'random_baselines_december.csv', index=False)
    prof = membership.groupby(['month', 'cluster']).agg(n=('territory_id', 'size'), support=('support', 'median'),
                total_rub=('total_rub', 'median'), total_yoy_pct=('total_yoy_pct', 'median'), **{f'share_{c}': (f'share_{c}', 'median') for c in SHARES}).reset_index()
    prof.to_csv(out / 'cluster_profiles.csv', index=False)
    # Summed observations describe the median municipality, not Russia's total spending.
    shifts = pd.DataFrame({'territory_id': ids, 'marketplace_share_2023': shares[:12, :, 2].mean(axis=0),
                           'marketplace_share_2024': shares[12:, :, 2].mean(axis=0),
                           'all_spending_growth_pct': 100 * (levels[12:, :, 0].mean(axis=0) / levels[:12, :, 0].mean(axis=0) - 1)})
    shifts['marketplace_share_change_pp'] = 100 * (shifts.marketplace_share_2024 - shifts.marketplace_share_2023)
    shifts.to_csv(out / 'annual_changes.csv', index=False)
    # One fixed PCA transform, fitted on development only, avoids map rotation.
    dev_x = np.concatenate([xx[months24.index(m)] for m in cfg['development_months']])
    pca = PCA(2).fit(dev_x)
    xy = np.stack([pca.transform(x) for x in xx])
    np.savez_compressed(out / 'analysis_arrays.npz', ids=ids, levels=levels, shares=shares, features=xx, labels=labels_array, support=support_array, xy=xy)
    final = {
        'primary_model': primary, 'overall_score_winner': best, 'n': len(ids), 'analysis_months': months24,
        'december_clusters': int(len(np.unique(december))), 'raw_switch_events': len(ev),
        'robust_switch_events': int(ev.robust.sum()), 'territories_with_robust_switch': int(ev.loc[ev.robust, 'territory_id'].nunique()),
        'december_median_support': float(np.median(support_array[-1])),
        'median_marketplace_share_change_pp': float(shifts.marketplace_share_change_pp.median()),
        'fraction_marketplace_share_increased': float((shifts.marketplace_share_change_pp > 0).mean()),
        'median_nominal_total_growth_pct': float(shifts.all_spending_growth_pct.median()),
        'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
        'external_validation': ext_summary, 'edge_distances': edge_summary,
        'elapsed_seconds': time.perf_counter() - start,
        'versions': {m: version(m) for m in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'pyarrow', 'igraph', 'leidenalg']},
        'python': platform.python_version(), 'config': cfg,
    }
    save_json(out / 'summary.json', final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.config)
