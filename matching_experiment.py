"""Two-candidate playlist matching, evaluated on held-out playlists."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'playlist_matching'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
CONTEXT_SIZE = 5
QUERIES_PER_PLAYLIST = 10
C_GRID = [0.01, 0.1, 1.0, 10.0]
AUDIO = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
         'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
FEATURES = ['closeness_' + x for x in AUDIO] + ['nearest_audio', 'artist_fraction']
MODEL_COLUMNS = {'Logistic: audio': list(range(11)),
                 'Logistic: audio + artist': list(range(12))}


def find_dataset():
    override = os.environ.get('SONGS620_PATH')
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'SONGS620_PATH does not exist: {path}')
        return path
    for path in [ROOT / 'songs620.csv', ROOT / 'data' / 'songs620.csv']:
        if path.is_file():
            return path
    raise FileNotFoundError('Put songs620.csv in data/ or set SONGS620_PATH.')


def prepare_data():
    OUT.mkdir(parents=True, exist_ok=True)
    path = find_dataset()
    raw = pd.read_csv(path)
    required = ['pid', 'track_uri', 'artist_uri'] + AUDIO
    if raw[required].isna().any().any():
        raise ValueError('Missing modeled metadata requires an explicit handling rule.')
    if not np.isfinite(raw[AUDIO].to_numpy()).all():
        raise ValueError('Nonfinite audio values.')
    metadata = raw.groupby('track_uri')[['artist_uri'] + AUDIO].nunique()
    if metadata.gt(1).any().any():
        raise ValueError('Conflicting metadata for a recording.')
    tracks = raw.drop_duplicates('track_uri').sort_values('track_uri').reset_index(drop=True)
    lookup = dict(zip(tracks.track_uri, tracks.index))
    raw['item'] = raw.track_uri.map(lookup)
    members = {int(pid): np.sort(g.item.unique()) for pid, g in raw.groupby('pid')}
    pids = np.random.default_rng(SEED).permutation(sorted(members))
    n_train, n_val = int(.8 * len(pids)), int(.1 * len(pids))
    split = {'train': pids[:n_train], 'validation': pids[n_train:n_train+n_val],
             'test': pids[n_train+n_val:]}
    train_items = np.unique(np.concatenate([members[int(p)] for p in split['train']]))
    # Each distinct training track contributes once to the audio scaler.
    audio = tracks[AUDIO].to_numpy(dtype=float)
    for col in ['tempo', 'duration_ms']:
        j = AUDIO.index(col)
        if (audio[:, j] < 0).any():
            raise ValueError(f'Negative {col}.')
        audio[:, j] = np.log1p(audio[:, j])
    scaler = StandardScaler().fit(audio[train_items])
    z = scaler.transform(audio)
    pop = np.zeros(len(tracks), dtype=int)
    for pid in split['train']:
        pop[members[int(pid)]] += 1
    split_frame = pd.DataFrame([(int(p), s) for s, ids in split.items() for p in ids],
                               columns=['pid', 'split'])
    split_frame.to_csv(OUT / 'playlist_split.csv', index=False)
    tracks.assign(item=tracks.index).to_csv(OUT / 'catalogue.csv', index=False)
    summary = {'rows': len(raw), 'unique_memberships': sum(map(len, members.values())),
               'tracks': len(tracks), 'playlists': len(members),
               'artists': int(tracks.artist_uri.nunique()),
               'repeated_memberships_removed': int(raw.duplicated(['pid', 'item']).sum()),
               'singleton_tracks': int(raw.groupby('item').pid.nunique().eq(1).sum()),
               'eligible_playlists': {s: sum(len(members[int(p)]) >= 6 for p in ids)
                                      for s, ids in split.items()},
               'split_playlists': {s: len(ids) for s, ids in split.items()},
               'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return dict(raw=raw, tracks=tracks, members=members, split=split, z=z,
                artists=tracks.artist_uri.to_numpy(), pop=pop, train_items=train_items,
                audio_scaler=scaler, summary=summary)


def make_queries(members, pids, catalogue_size, seed, per_playlist=QUERIES_PER_PLAYLIST):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in sorted(pids):
        playlist = members[int(pid)]
        if len(playlist) < CONTEXT_SIZE + 1:
            continue
        outsiders = np.setdiff1d(np.arange(catalogue_size), playlist, assume_unique=True)
        if not len(outsiders):
            raise ValueError('No nonmember candidates.')
        for q in range(per_playlist):
            six = rng.choice(playlist, CONTEXT_SIZE + 1, replace=False)
            positive, negative = int(six[-1]), int(rng.choice(outsiders))
            label = int(rng.integers(2))  # 1 means candidate A is the member.
            a, b = (positive, negative) if label else (negative, positive)
            rows.append([int(pid), q, *map(int, six[:5]), a, b, label])
    return pd.DataFrame(rows, columns=['pid', 'query', 'h0', 'h1', 'h2', 'h3', 'h4',
                                       'candidate_a', 'candidate_b', 'label'])


def candidate_features(data, contexts, candidates):
    seeds = data['z'][contexts]
    candidate = data['z'][candidates]
    centroid = seeds.mean(axis=1)
    closeness = -np.abs(candidate - centroid)
    nearest = -np.sqrt(((seeds - candidate[:, None, :]) ** 2).sum(axis=2)).min(axis=1)
    artist_fraction = (data['artists'][contexts] == data['artists'][candidates, None]).mean(axis=1)
    return np.column_stack([closeness, nearest, artist_fraction])


def query_features(data, queries):
    contexts = queries[[f'h{i}' for i in range(5)]].to_numpy(dtype=int)
    a = candidate_features(data, contexts, queries.candidate_a.to_numpy())
    b = candidate_features(data, contexts, queries.candidate_b.to_numpy())
    return a - b


def baseline_scores(data, queries, delta):
    # Greater is always stronger evidence for candidate A.
    return {'Random (expected)': np.zeros(len(queries)),
            'Popularity': data['pop'][queries.candidate_a] - data['pop'][queries.candidate_b],
            'Artist overlap': delta[:, -1],
            'Audio distance': delta[:, :10].sum(axis=1)}


def correct_credit(scores, labels):
    """Fair random tie-breaking has expected correctness 0.5."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    return np.where(scores == 0, .5, (scores > 0) == labels).astype(float)


def fit_models(train_x, train_y, val_x, val_y):
    fitted, grid, selected = {}, [], {}
    for name, columns in MODEL_COLUMNS.items():
        # Mirrored pairs prevent candidate-position bias and enforce swap symmetry.
        x = np.vstack([train_x[:, columns], -train_x[:, columns]])
        y = np.concatenate([train_y, 1 - train_y])
        options = []
        for c in C_GRID:
            model = make_pipeline(StandardScaler(with_mean=False),
                                  LogisticRegression(C=c, fit_intercept=False, max_iter=2000))
            model.fit(x, y)
            accuracy = correct_credit(model.decision_function(val_x[:, columns]), val_y).mean()
            grid.append({'model': name, 'C': c, 'validation_accuracy': accuracy})
            options.append((accuracy, -c, model))
        accuracy, neg_c, model = max(options, key=lambda v: (v[0], v[1]))
        fitted[name] = model
        selected[name] = {'C': -neg_c, 'validation_accuracy': accuracy}
    primary = max(selected, key=lambda name: (selected[name]['validation_accuracy'],
                                             -len(MODEL_COLUMNS[name])))
    return fitted, pd.DataFrame(grid), selected, primary


def bootstrap_interval(values, seed=SEED, draws=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    averages = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(averages, [.025, .975])]


def evaluate(data, queries, delta, models):
    scores = baseline_scores(data, queries, delta)
    scores.update({name: model.decision_function(delta[:, MODEL_COLUMNS[name]])
                   for name, model in models.items()})
    details, metrics = [], []
    for name, score in scores.items():
        d = queries.copy()
        d['model'], d['score_a_minus_b'] = name, score
        d['correct_credit'] = correct_credit(score, queries.label)
        d['tie'] = np.asarray(score) == 0
        positive = np.where(queries.label, queries.candidate_a, queries.candidate_b)
        d['member_seen_in_training'] = data['pop'][positive] > 0
        per_playlist = d.groupby('pid').correct_credit.mean()
        low, high = bootstrap_interval(per_playlist)
        metrics.append({'model': name, 'accuracy': per_playlist.mean(),
                        'ci_low': low, 'ci_high': high, 'tie_fraction': d.tie.mean(),
                        'queries': len(d), 'playlists': len(per_playlist)})
        details.append(d)
    return pd.DataFrame(metrics), pd.concat(details, ignore_index=True)


def paired_comparison(details, first, second):
    table = details.groupby(['pid', 'model']).correct_credit.mean().unstack('model')
    delta = table[first] - table[second]
    low, high = bootstrap_interval(delta)
    return {'first': first, 'second': second, 'difference': float(delta.mean()),
            'ci_low': low, 'ci_high': high, 'playlists': len(delta), 'bootstrap_draws': 10000}


def run_experiment(data=None):
    if data is None:
        data = prepare_data()
    queries = {s: make_queries(data['members'], ids, len(data['tracks']), SEED + i + 1)
               for i, (s, ids) in enumerate(data['split'].items())}
    for name, q in queries.items():
        q.to_csv(OUT / f'{name}_queries.csv', index=False)
    delta = {s: query_features(data, q) for s, q in queries.items()}
    models, grid, selection, primary = fit_models(delta['train'], queries['train'].label.to_numpy(),
                                                  delta['validation'], queries['validation'].label.to_numpy())
    grid.to_csv(OUT / 'validation_grid.csv', index=False)
    frozen = {'models': selection, 'primary_model': primary, 'seed': SEED,
              'context_size': CONTEXT_SIZE, 'queries_per_playlist': QUERIES_PER_PLAYLIST,
              'negative_sampling': 'uniform distinct track outside the complete target playlist'}
    (OUT / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    # Test scores are accessed only after validation selection is frozen.
    metrics, details = evaluate(data, queries['test'], delta['test'], models)
    metrics.to_csv(OUT / 'test_metrics.csv', index=False)
    details.to_csv(OUT / 'test_per_query.csv', index=False)
    coefficients = []
    for name, model in models.items():
        for feature, weight in zip(np.array(FEATURES)[MODEL_COLUMNS[name]], model[-1].coef_[0]):
            coefficients.append({'model': name, 'feature': feature, 'coefficient': weight})
    coefficients = pd.DataFrame(coefficients)
    coefficients.to_csv(OUT / 'coefficients.csv', index=False)
    subgroup = details.groupby(['model', 'member_seen_in_training', 'pid']).correct_credit.mean()
    subgroup = subgroup.groupby(level=[0, 1]).agg(['mean', 'count']).reset_index()
    subgroup.columns = ['model', 'member_seen_in_training', 'accuracy', 'playlists_with_queries']
    subgroup.to_csv(OUT / 'support_subgroups.csv', index=False)
    comparisons = [paired_comparison(details, primary, 'Random (expected)'),
                   paired_comparison(details, primary, 'Audio distance'),
                   paired_comparison(details, 'Logistic: audio + artist', 'Logistic: audio')]
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_comparisons.csv', index=False)
    summary = {'data': data['summary'], 'selection': frozen,
               'query_counts': {s: len(q) for s, q in queries.items()},
               'comparisons': comparisons,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                               'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    (OUT / 'experiment_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = metrics.iloc[::-1]
    ax.barh(ordered.model, ordered.accuracy, color=['#238b45' if n == primary else '#4786b8'
                                                  for n in ordered.model])
    ax.errorbar(ordered.accuracy, np.arange(len(ordered)),
                xerr=np.vstack([ordered.accuracy-ordered.ci_low, ordered.ci_high-ordered.accuracy]),
                fmt='none', ecolor='black', capsize=3)
    ax.axvline(.5, color='black', linestyle='--', label='Random: 50%')
    ax.set(xlim=(0, 1), xlabel='Two-candidate accuracy (95% playlist bootstrap interval)',
           title='Which song fits this playlist?')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'accuracy.png', dpi=160)
    plt.close(fig)
    return dict(data=data, queries=queries, models=models, grid=grid, metrics=metrics,
                details=details, coefficients=coefficients, subgroups=subgroup, summary=summary)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        results = run_experiment()
    print(results['metrics'].to_string(index=False))
