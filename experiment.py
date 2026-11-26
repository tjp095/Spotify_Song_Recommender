"""Reproducible playlist-completion experiment for Assignment 2, parts 2-4.

All model selection uses validation interactions. The final comparison uses
previously untouched test interactions after refitting on train + validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SEED = 42
K = 20
NUMERIC = ["danceability", "energy", "loudness", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
CORE = ["danceability", "energy", "loudness", "valence", "tempo"]


def load_data():
    raw = pd.read_csv(ROOT / "data/songs620.csv")
    assert raw[["pid", "track_uri"]].notna().all().all()
    edges = raw.drop_duplicates(["pid", "track_uri"]).copy()
    # Stable vocabulary from static catalogue metadata, without interaction counts.
    tracks = raw.drop_duplicates("track_uri").sort_values("track_uri").reset_index(drop=True)
    pids = np.sort(edges.pid.unique())
    track_index = {s: i for i, s in enumerate(tracks.track_uri)}
    pid_index = {p: i for i, p in enumerate(pids)}
    edges["u"] = edges.pid.map(pid_index)
    edges["i"] = edges.track_uri.map(track_index)
    return raw, edges, tracks, pids


def split_edges(edges, shape):
    rng = np.random.default_rng(SEED)
    rows = {s: [] for s in ["train", "validation", "test"]}
    for u, group in edges.groupby("u", sort=True):
        items = rng.permutation(np.sort(group.i.to_numpy()))
        assert len(items) >= 5
        # Floor each holdout at 20%; remainder belongs to train.
        h = max(1, int(len(items) * .2))
        for name, values in zip(rows, [items[2*h:], items[:h], items[h:2*h]]):
            rows[name].extend((u, int(i)) for i in values)
    matrices = {}
    for name, pairs in rows.items():
        u, i = np.array(pairs).T
        matrices[name] = csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=shape)
    a, b, c = matrices.values()
    assert a.multiply(b).nnz == a.multiply(c).nnz == b.multiply(c).nnz == 0
    assert a.nnz + b.nnz + c.nnz == len(edges)
    assert all(np.all(x.getnnz(axis=1) > 0) for x in matrices.values())
    manifest = pd.concat([pd.DataFrame(pairs, columns=["u", "i"]).assign(split=name)
                          for name, pairs in rows.items()], ignore_index=True)
    manifest.to_csv(OUT / "split_manifest.csv", index=False)
    return matrices


def audio_features(tracks, observed, representation):
    cols = CORE if representation == "core" else NUMERIC
    x = tracks[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Avoid letting an extreme duration dominate the geometry.
    if "duration_ms" in cols:
        x["duration_ms"] = np.log1p(x["duration_ms"].clip(lower=0))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_fit = imputer.fit_transform(x.loc[observed])
    scaler = StandardScaler().fit(x_fit)
    z = scaler.transform(imputer.transform(x)).astype(np.float32)
    if representation == "full":
        # Fixed semantic categories: no category order is assumed.
        cats = []
        for col, values in [("key", range(-1, 12)), ("mode", [-1, 0, 1]),
                            ("time_signature", range(-1, 8))]:
            v = pd.to_numeric(tracks[col], errors="coerce").fillna(-1).to_numpy()
            cats.extend((v == value).astype(np.float32)[:, None] for value in values)
        z = np.hstack([z] + cats)
    return normalize(z).astype(np.float32)


class Ranker:
    """Small score interface; no test labels are accepted by this class."""
    def __init__(self, name, parameters=None):
        self.name, self.parameters = name, parameters or {}

    def fit(self, interactions, tracks):
        start = perf_counter()
        self.b = interactions.tocsr().astype(np.float32)
        self.pop = np.asarray(self.b.sum(axis=0)).ravel()
        self.n_items = self.b.shape[1]
        if self.name == "Playlist-kNN":
            b_norm = normalize(self.b)
            similarity = (b_norm @ b_norm.T).toarray()
            np.fill_diagonal(similarity, 0)  # Never use a playlist as its own neighbor.
            k = self.parameters["neighbors"]
            neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
            u = np.repeat(np.arange(self.b.shape[0]), k)
            v = neighbors.ravel()
            weights = similarity[u, v] ** self.parameters["power"]
            self.w = csr_matrix((weights, (u, v)), shape=similarity.shape)
        elif self.name == "PureSVD":
            self.svd = TruncatedSVD(n_components=self.parameters["rank"], n_iter=7,
                                    random_state=SEED)
            self.embedding = self.svd.fit_transform(self.b)
        elif self.name == "Audio centroid":
            self.features = audio_features(tracks, self.pop > 0, self.parameters["features"])
            self.profiles = normalize(np.asarray(self.b @ self.features))
        elif self.name != "Popularity":
            raise ValueError(self.name)
        self.fit_seconds = perf_counter() - start
        return self

    def scores(self, first, last):
        if self.name == "Popularity":
            return np.broadcast_to(self.pop, (last-first, self.n_items)).copy()
        if self.name == "Playlist-kNN":
            return (self.w[first:last] @ self.b).toarray()
        if self.name == "PureSVD":
            return self.embedding[first:last] @ self.svd.components_
        return self.profiles[first:last] @ self.features.T


def top_items(scores, seen, popularity, k=50):
    """Exact top-k, with deterministic popularity then item-ID tie breaks."""
    scores = np.asarray(scores).copy()
    scores[seen] = -np.inf
    assert np.isfinite(scores[~seen]).all()
    cutoff = np.partition(scores, len(scores)-k)[len(scores)-k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    tied = tied[np.lexsort((tied, -popularity[tied]))][:k-len(above)]
    selected = np.r_[above, tied]
    return selected[np.lexsort((selected, -popularity[selected], -scores[selected]))]


def recommend(model, max_k=50):
    start = perf_counter()
    ranked = []
    for first in range(0, model.b.shape[0], 32):
        last = min(first+32, model.b.shape[0])
        scores = model.scores(first, last)
        for offset, u in enumerate(range(first, last)):
            seen = np.zeros(model.n_items, dtype=bool)
            seen[model.b[u].indices] = True
            top = top_items(scores[offset], seen, model.pop, max_k)
            assert len(np.unique(top)) == max_k and not seen[top].any()
            ranked.append(top)
    return np.array(ranked), perf_counter()-start


def evaluate(ranked, target, fit_matrix, tracks):
    rows = []
    pop = np.asarray(fit_matrix.sum(axis=0)).ravel()
    artists = tracks.artist_uri.fillna("unknown").to_numpy()
    discounts = 1 / np.log2(np.arange(2, 52))
    for u, recs in enumerate(ranked):
        truth = target[u].indices
        hits = np.isin(recs, truth)
        row = {"u": u, "target_count": len(truth), "seed_count": fit_matrix[u].nnz}
        for k in [10, 20, 50]:
            h = hits[:k].sum()
            row.update({f"Recall@{k}": h/len(truth), f"Precision@{k}": h/k,
                        f"NDCG@{k}": float(hits[:k] @ discounts[:k]) / discounts[:min(k,len(truth))].sum(),
                        f"HitRate@{k}": float(h > 0)})
        warm, cold = truth[pop[truth] > 0], truth[pop[truth] == 0]
        row["WarmRecall@20"] = np.isin(recs[:20], warm).sum()/len(warm) if len(warm) else np.nan
        row["ColdRecall@20"] = np.isin(recs[:20], cold).sum()/len(cold) if len(cold) else np.nan
        row["cold_targets"] = len(cold)
        row["Novelty@20"] = float(np.mean(np.log2((fit_matrix.shape[0]+1)/(pop[recs[:20]]+1))))
        row["ArtistDiversity@20"] = len(np.unique(artists[recs[:20]]))/20
        rows.append(row)
    return pd.DataFrame(rows)


def paired_uncertainty(a, b):
    """Percentile paired bootstrap + two-sided paired sign-flip test."""
    rng = np.random.default_rng(2026)
    d = np.asarray(a) - np.asarray(b)
    boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    extreme = 0
    for _ in range(20):
        perm = (rng.choice([-1, 1], size=(500, len(d))) * d).mean(axis=1)
        extreme += np.count_nonzero(np.abs(perm) >= abs(d.mean()) - 1e-15)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(d.mean()), float(lo), float(hi), (extreme+1)/10001


def run_eda(raw, edges, tracks, train):
    sizes = edges.groupby("pid").size()
    freq = edges.groupby("track_uri").size().sort_values(ascending=False)
    summary = {
        "raw_rows": len(raw), "unique_interactions": len(edges),
        "duplicate_playlist_track_rows": len(raw)-len(edges),
        "playlists": edges.pid.nunique(), "tracks": len(tracks),
        "artists": tracks.artist_uri.nunique(),
        "density": len(edges)/(edges.pid.nunique()*len(tracks)),
        "singleton_tracks": int((freq == 1).sum()),
        "singleton_track_fraction": float((freq == 1).mean()),
        "top_1pct_interaction_share": float(freq.iloc[:int(np.ceil(.01*len(freq)))].sum()/len(edges)),
        "playlist_size": {k: float(v) for k,v in sizes.describe().items()},
        "missing_by_column": {k:int(v) for k,v in raw.isna().sum().items()},
        "pid_min": int(edges.pid.min()), "pid_max": int(edges.pid.max()),
        "metadata_conflicting_tracks": int((raw.groupby("track_uri")[NUMERIC + ["key", "mode", "time_signature", "artist_uri"]].nunique(dropna=False).max(axis=1)>1).sum()),
    }
    summary["invalid_numeric_values"] = {}
    for col in NUMERIC:
        v = pd.to_numeric(tracks[col], errors="coerce")
        summary["invalid_numeric_values"][col] = int((~np.isfinite(v)).sum())
    tracks[NUMERIC].describe().T.to_csv(OUT / "audio_summary.csv")
    pd.DataFrame({"column": raw.columns, "missing": raw.isna().sum().to_numpy()}).to_csv(OUT / "missingness.csv", index=False)
    raw.groupby("track_uri").size().rename("row_count").to_csv(OUT / "track_frequencies.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1,3,figsize=(14,3.7), layout="constrained")
    ax[0].hist(sizes, bins=25, color="#237a69")
    ax[0].set(xlabel="Unique tracks per playlist", ylabel="Playlists", title="Playlist length")
    ax[1].loglog(np.arange(1,len(freq)+1), freq.values, color="#386ba8")
    ax[1].set(xlabel="Track popularity rank (log)", ylabel="Playlist count (log)", title="Long-tail interactions")
    corr = tracks[["danceability","energy","loudness","acousticness","valence","tempo"]].corr()
    im = ax[2].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[2].set(xticks=range(6), yticks=range(6), yticklabels=corr.columns, title="Track-level correlations")
    ax[2].set_xticklabels(corr.columns, rotation=65, ha="right")
    fig.colorbar(im, ax=ax[2], shrink=.8)
    fig.savefig(OUT / "eda.png", dpi=160)
    plt.close(fig)
    # Training-only evidence for feature relevance; one centroid does not presume
    # that every playlist is acoustically homogeneous.
    z = audio_features(tracks, np.asarray(train.sum(axis=0)).ravel()>0, "core")
    rng = np.random.default_rng(SEED)
    within, random = [], []
    for u in range(train.shape[0]):
        ids = train[u].indices
        n = min(20, len(ids))
        for _ in range(n):
            a,b = rng.choice(ids, 2, replace=False)
            within.append(float(z[a] @ z[b]))
            a,b = rng.choice(len(tracks),2,replace=False)
            random.append(float(z[a] @ z[b]))
    summary["training_audio_cosine_within_mean"] = float(np.mean(within))
    summary["catalogue_audio_cosine_random_mean"] = float(np.mean(random))
    summary["audio_energy_loudness_correlation"] = float(tracks.energy.corr(tracks.loudness))
    fig,ax = plt.subplots(figsize=(7,3.2), layout="constrained")
    ax.hist(within,bins=40,density=True,alpha=.6,label="Within training playlist")
    ax.hist(random,bins=40,density=True,alpha=.5,label="Random catalogue pair")
    ax.set(xlabel="Cosine similarity of standardized core audio features", ylabel="Density", title="Do audio features reflect playlist coherence?")
    ax.legend()
    fig.savefig(OUT / "audio_coherence.png",dpi=160)
    plt.close(fig)
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run():
    OUT.mkdir(exist_ok=True)
    raw, edges, tracks, pids = load_data()
    shape = (len(pids),len(tracks))
    matrices = split_edges(edges, shape)
    train, validation, test = matrices.values()
    tracks[["track_uri","artist_uri","track_name","artist_name"]].to_csv(OUT / "item_vocabulary.csv",index=False)
    pd.DataFrame({"u": np.arange(len(pids)), "pid":pids}).to_csv(OUT / "playlist_vocabulary.csv",index=False)
    summary = run_eda(raw, edges, tracks, train)
    print("EDA:", json.dumps({k:summary[k] for k in ["raw_rows","unique_interactions","tracks","singleton_track_fraction"]}),flush=True)
    configs = [("Popularity", {})]
    configs += [("Playlist-kNN", {"neighbors":k, "power":p}) for k in [20,50,100] for p in [1,2]]
    configs += [("PureSVD", {"rank":r}) for r in [16,32,64]]
    configs += [("Audio centroid", {"features":f}) for f in ["core","full"]]
    validation_rows = []
    for name,parameters in configs:
        model = Ranker(name,parameters).fit(train, tracks)
        ranked, seconds = recommend(model)
        metrics = evaluate(ranked,validation,train,tracks)
        row = {"model":name,"parameters":json.dumps(parameters,sort_keys=True),
               "fit_seconds":model.fit_seconds,"rank_seconds":seconds,
               **{k:float(metrics[k].mean()) for k in ["NDCG@20","Recall@20","Precision@20"]}}
        validation_rows.append(row)
        print("Validation:", json.dumps(row),flush=True)
    val_table = pd.DataFrame(validation_rows)
    val_table.to_csv(OUT / "validation_results.csv",index=False)
    # Stable tie: earlier configuration in the declared grid wins.
    selected = {name: group.loc[group["NDCG@20"].idxmax()].to_dict()
                for name,group in val_table.groupby("model",sort=False)}
    winner = max(selected, key=lambda n:selected[n]["NDCG@20"])
    # Test data are first used here, after the full selection is frozen.
    fit_matrix = (train + validation).tocsr()
    final_rows, all_per_playlist, ranks = [], {}, {}
    for name,selection in selected.items():
        model = Ranker(name,json.loads(selection["parameters"])).fit(fit_matrix,tracks)
        ranked,seconds = recommend(model)
        metrics = evaluate(ranked,test,fit_matrix,tracks)
        ranks[name], all_per_playlist[name] = ranked, metrics
        metrics.assign(model=name).to_csv(OUT / (name.lower().replace(" ","_")+"_per_playlist.csv"),index=False)
        np.save(OUT / (name.lower().replace(" ","_")+"_top50.npy"),ranked)
        row = {"model":name, "parameters":selection["parameters"],
               **{c:float(metrics[c].mean()) for c in metrics if "@" in c},
               "Coverage@20":len(np.unique(ranked[:,:20]))/shape[1],
               "fit_seconds":model.fit_seconds,"rank_seconds":seconds}
        final_rows.append(row)
        print("Test:",json.dumps(row),flush=True)
    result_table = pd.DataFrame(final_rows)
    result_table.to_csv(OUT / "test_results.csv",index=False)
    base = all_per_playlist["Popularity"]["NDCG@20"]
    uncertainty = []
    for name in selected:
        if name == "Popularity": continue
        delta,lo,hi,p = paired_uncertainty(all_per_playlist[name]["NDCG@20"],base)
        uncertainty.append({"model":name,"delta_ndcg20_vs_popularity":delta,
                            "ci_low":lo,"ci_high":hi,"p_two_sided":p})
    uncertainty = pd.DataFrame(uncertainty)
    order = np.argsort(uncertainty.p_two_sided.to_numpy())
    adjusted = np.maximum.accumulate(np.minimum(1,uncertainty.p_two_sided.to_numpy()[order] * np.arange(len(order),0,-1)))
    uncertainty.loc[order,"p_holm"] = adjusted
    uncertainty.to_csv(OUT / "paired_comparisons.csv",index=False)
    # Confidence interval on each model mean, resampling playlists as the unit.
    rng = np.random.default_rng(2026)
    indices = rng.integers(0,shape[0],size=(2000,shape[0]))
    ci = {}
    for name,m in all_per_playlist.items():
        ci[name] = np.quantile(m["NDCG@20"].to_numpy()[indices].mean(axis=1),[.025,.975]).tolist()
    length = np.asarray(fit_matrix.sum(axis=1)).ravel()
    # Fixed cutpoints give transparent groups and avoid ambiguous quantile ties.
    bins = pd.cut(length, bins=[0,20,50,np.inf],labels=["1-20 seeds","21-50 seeds","51+ seeds"])
    subgroup = []
    for name,m in all_per_playlist.items():
        g=m.assign(length_group=bins).groupby("length_group",observed=True).agg(
            playlists=("u","size"),ndcg20=("NDCG@20","mean"),recall20=("Recall@20","mean"))
        subgroup.append(g.reset_index().assign(model=name))
    pd.concat(subgroup).to_csv(OUT / "length_subgroups.csv",index=False)
    example_u = 0  # Chosen in advance, never cherry-picked by performance.
    example_rows=[]
    for name,ranked in ranks.items():
        for rank,i in enumerate(ranked[example_u,:10],1):
            example_rows.append({"model":name,"rank":rank,"track":tracks.iloc[i].track_name,
                                 "artist":tracks.iloc[i].artist_name,"test_hit":int(i in test[example_u].indices)})
    pd.DataFrame(example_rows).to_csv(OUT / "example_recommendations.csv",index=False)
    tracks.iloc[fit_matrix[example_u].indices][["track_name","artist_name"]].to_csv(OUT / "example_seeds.csv",index=False)
    experiment_summary = {
        "seed":SEED,"primary_metric":"NDCG@20", "selection_winner":winner,
        "split_counts":{k:int(m.nnz) for k,m in matrices.items()},
        "cold_test_interactions":int(all_per_playlist["Popularity"].cold_targets.sum()),
        "test_interactions":int(test.nnz),"test_catalogue_size":shape[1],
        "ndcg20_mean_confidence_intervals":ci,
        "dataset_sha256":hashlib.sha256((ROOT / "data/songs620.csv").read_bytes()).hexdigest(),
        "environment":{"python":platform.python_version(),"numpy":np.__version__,
                       "pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
        "selected":selected,
    }
    (OUT / "experiment_summary.json").write_text(json.dumps(experiment_summary,indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout="constrained")
    names=result_table.model.to_list()
    for ax,metric in zip(axes,["NDCG@20","Recall@20","Coverage@20"]):
        values=result_table[metric].to_numpy()
        ax.bar(np.arange(len(names)),values,color=["#a6adb5","#287d6b","#386ba8","#c58c39"])
        if metric=="NDCG@20":
            lower=np.array([ci[n][0] for n in names]); upper=np.array([ci[n][1] for n in names])
            ax.errorbar(np.arange(len(names)),values,yerr=np.vstack([values-lower,upper-values]),fmt="none",color="black",capsize=3)
        ax.set(xticks=np.arange(len(names)),title=metric,ylabel="Mean" if metric!="Coverage@20" else "Catalogue fraction")
        ax.set_xticklabels(names,rotation=25,ha="right")
    fig.savefig(OUT / "model_comparison.png",dpi=170)
    plt.close(fig)
    print("Finished. Validation-selected winner:",winner,flush=True)
    return experiment_summary


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
