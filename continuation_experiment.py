"""Five ordered tracks -> the sixth track, with playlist-disjoint evaluation.

Run this file or the self-contained assignment_2.ipynb. No HTML is exported.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "next_song"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, issparse
from scipy.stats import binomtest
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

SEED = 42
HISTORY_LENGTH = 5
NEGATIVES = 10
REGULARIZATION = [0.01, 0.1, 1.0, 10.0]
AUDIO_COLUMNS = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                 "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
HISTORY_NAMES = ["log_track_support", "cosine_mean", "cosine_max", "cosine_last",
                 "log_markov_last", "log_markov_mean", "repeat_fraction"]
ARTIST_NAMES = ["same_last_artist", "history_artist_fraction", "log_artist_support",
                "log_artist_markov_last", "log_artist_markov_mean"]
AUDIO_NAMES = ["audio_cosine_last", "audio_cosine_centroid", "audio_distance_last",
               "audio_distance_centroid"] + ["audio_last_abs_"+c for c in AUDIO_COLUMNS]
FEATURE_NAMES = HISTORY_NAMES + ARTIST_NAMES + AUDIO_NAMES
MODEL_FEATURES = {
    "History": list(range(len(HISTORY_NAMES))),
    "History + artist": list(range(len(HISTORY_NAMES)+len(ARTIST_NAMES))),
    "History + artist + audio": list(range(len(FEATURE_NAMES))),
}
BASELINES = ["Popularity", "Last artist", "Item cosine", "Markov", "Audio similarity"]


def find_dataset():
    override = os.environ.get("SONGS620_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SONGS620_PATH does not exist: {path}")
        return path.resolve()
    for path in [ROOT / "songs620.csv", ROOT / "data" / "songs620.csv",
                 Path.home() / "Desktop" / "songs620.csv"]:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Put songs620.csv beside the notebook or set SONGS620_PATH.")


def queries_from_sequences(sequences, pids, start=0, all_windows=False):
    rows = []
    for pid in pids:
        seq = sequences[int(pid)]
        starts = range(len(seq)-5) if all_windows else [start]
        for pos in starts:
            if pos+5 < len(seq):
                rows.append([int(pid), pos, *seq[pos:pos+5], seq[pos+5]])
    return pd.DataFrame(rows, columns=["pid", "start", "h0", "h1", "h2", "h3", "h4", "target"])


def prepare_data():
    OUT.mkdir(parents=True, exist_ok=True)
    path = find_dataset()
    raw = pd.read_csv(path)
    assert raw[["pid", "pos", "track_uri", "artist_uri"]].notna().all().all()
    assert not raw.duplicated(["pid", "pos"]).any(), "Ambiguous playlist positions."
    assert (raw.pos == raw.pos.astype(int)).all()
    raw = raw.sort_values(["pid", "pos"], kind="stable").reset_index(drop=True)
    for pid, group in raw.groupby("pid", sort=True):
        assert np.array_equal(group.pos.to_numpy(), np.arange(len(group))), f"Gaps in playlist {pid}"
    metadata_cols = AUDIO_COLUMNS + ["key", "mode", "time_signature", "artist_uri"]
    conflicts = raw.groupby("track_uri")[metadata_cols].nunique(dropna=False).max(axis=1)
    assert not (conflicts > 1).any(), "Resolve conflicting track metadata before fitting."
    tracks = raw.drop_duplicates("track_uri").sort_values("track_uri").reset_index(drop=True)
    item_index = {uri:i for i,uri in enumerate(tracks.track_uri)}
    artists = np.sort(tracks.artist_uri.unique())
    artist_index = {uri:i for i,uri in enumerate(artists)}
    artist_ids = tracks.artist_uri.map(artist_index).to_numpy()
    raw["item"] = raw.track_uri.map(item_index)
    sequences = {int(pid):g.item.to_numpy(dtype=np.int32) for pid,g in raw.groupby("pid",sort=True)}
    pids = np.sort(raw.pid.unique())
    assert len(pids) == 1000, "The declared 800/100/100 design expects this 1,000-playlist file."
    shuffled = np.random.default_rng(SEED).permutation(pids)
    split = {"train":shuffled[:800], "validation":shuffled[800:900], "test":shuffled[900:]}
    manifest = pd.concat([pd.DataFrame({"pid":ids,"split":name}) for name,ids in split.items()])
    assert manifest.pid.nunique() == len(pids)
    manifest.to_csv(OUT / "playlist_split.csv",index=False)
    queries = {s:queries_from_sequences(sequences,ids,all_windows=(s=="train")) for s,ids in split.items()}
    for name,q in queries.items(): q.to_csv(OUT / (name+"_queries.csv"),index=False)
    tracks[["track_uri", "artist_uri", "track_name", "artist_name"]].to_csv(OUT / "catalogue.csv",index_label="item")
    train_rows = raw[raw.pid.isin(split["train"])]
    observed = np.zeros(len(tracks),dtype=bool)
    observed[train_rows.item.unique()] = True
    audio = tracks[AUDIO_COLUMNS].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).copy()
    audio["duration_ms"] = np.log1p(audio.duration_ms.clip(lower=0))
    imputer = SimpleImputer(strategy="median",keep_empty_features=True).fit(audio.loc[observed])
    scaler = StandardScaler().fit(imputer.transform(audio.loc[observed]))
    z = scaler.transform(imputer.transform(audio)).astype(np.float32)
    zn = normalize(z).astype(np.float32)
    sizes=raw.groupby("pid").size()
    unique_memberships=raw.drop_duplicates(["pid","track_uri"])
    counts=unique_memberships.groupby("track_uri").size()
    successor_same_artist=[]
    within_audio=[]
    random_audio=[]
    rng=np.random.default_rng(4242)
    for pid in split["train"]:
        seq=sequences[int(pid)]
        successor_same_artist.extend(artist_ids[seq[:-1]]==artist_ids[seq[1:]])
        within_audio.extend(np.sum(zn[seq[:-1]]*zn[seq[1:]],axis=1))
        a=rng.integers(0,len(tracks),len(seq)-1)
        b=rng.integers(0,len(tracks),len(seq)-1)
        random_audio.extend(np.sum(zn[a]*zn[b],axis=1))
    summary={"rows":len(raw),"playlists":len(pids),"tracks":len(tracks),"artists":len(artists),
             "repeat_memberships":int(raw.duplicated(["pid","track_uri"]).sum()),
             "unique_memberships":len(unique_memberships),"density":len(unique_memberships)/(len(pids)*len(tracks)),
             "missing_by_column":{c:int(n) for c,n in raw.isna().sum().items()},
             "nonfinite_audio_values":int((~np.isfinite(tracks[AUDIO_COLUMNS].to_numpy(dtype=float))).sum()),
             "playlist_size":{k:float(v) for k,v in sizes.describe().items()},
             "singleton_tracks":int((counts==1).sum()),
             "train_same_artist_transition_fraction":float(np.mean(successor_same_artist)),
             "train_adjacent_audio_cosine":float(np.mean(within_audio)),
             "random_audio_cosine":float(np.mean(random_audio)),
             "query_counts":{s:len(q) for s,q in queries.items()},
             "excluded_short_openings":{s:len(ids)-len(queries_from_sequences(sequences,ids)) for s,ids in split.items()},
             "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "data_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    tracks[AUDIO_COLUMNS].describe().T.to_csv(OUT / "audio_summary.csv")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,3,figsize=(13,3.5),layout="constrained")
    ax[0].hist(sizes,bins=25,color="#297b69")
    ax[0].set(title="Playlist lengths",xlabel="Ordered tracks",ylabel="Playlists")
    freq=np.sort(counts.to_numpy())[::-1]
    ax[1].loglog(np.arange(1,len(freq)+1),freq,color="#3979ad")
    ax[1].set(title="Track support has a long tail",xlabel="Popularity rank (log)",ylabel="Playlist support (log)")
    ax[2].hist(within_audio,bins=40,alpha=.6,density=True,label="Training adjacent pairs")
    ax[2].hist(random_audio,bins=40,alpha=.5,density=True,label="Random catalogue pairs")
    ax[2].set(title="Audio continuity",xlabel="Standardized audio cosine",ylabel="Density")
    ax[2].legend(fontsize=8)
    fig.savefig(OUT / "data_overview.png",dpi=160); plt.close(fig)
    return {"raw":raw,"tracks":tracks,"sequences":sequences,"split":split,"queries":queries,
            "artist_ids":artist_ids,"n_artists":len(artists),"z":z,"zn":zn,"summary":summary}


class TrainingStatistics:
    """Interaction statistics from an explicitly named set of training playlists."""
    def __init__(self,data,pids):
        self.pids=frozenset(int(p) for p in pids)
        self.n_items=len(data["tracks"])
        self.n_artists=data["n_artists"]
        self.artist_ids=data["artist_ids"]
        u=[]; items=[]; source=[]; target=[]
        for row,pid in enumerate(sorted(self.pids)):
            seq=data["sequences"][pid]
            unique=np.unique(seq)
            u.extend([row]*len(unique)); items.extend(unique)
            source.extend(seq[:-1]); target.extend(seq[1:])
        b=csr_matrix((np.ones(len(items),dtype=np.float32),(u,items)),shape=(len(self.pids),self.n_items))
        self.pop=np.asarray(b.sum(axis=0)).ravel()
        self.prior=(self.pop+1)/(self.pop.sum()+self.n_items)
        similarity=(b.T @ b).tocsr()
        denom=np.sqrt(np.maximum(self.pop,1))
        self.cosine=similarity.multiply(1/denom[:,None]).multiply(1/denom[None,:]).tocsr()
        self.transition=csr_matrix((np.ones(len(source),dtype=np.float32),(source,target)),shape=(self.n_items,self.n_items))
        self.transition_out=np.asarray(self.transition.sum(axis=1)).ravel()
        au=self.artist_ids[np.asarray(items)]
        ba=csr_matrix((np.ones(len(u),dtype=np.float32),(u,au)),shape=(len(self.pids),self.n_artists))
        ba.data[:]=1
        self.artist_pop=np.asarray(ba.sum(axis=0)).ravel()
        self.artist_prior=(self.artist_pop+1)/(self.artist_pop.sum()+self.n_artists)
        aa=self.artist_ids[np.asarray(source)]; ab=self.artist_ids[np.asarray(target)]
        self.artist_transition=csr_matrix((np.ones(len(aa),dtype=np.float32),(aa,ab)),shape=(self.n_artists,self.n_artists))
        self.artist_out=np.asarray(self.artist_transition.sum(axis=1)).ravel()


def sparse_lookup(matrix,rows,columns):
    values=matrix[rows,columns]
    return values.toarray() if issparse(values) else np.asarray(values)


def candidate_features(data,stats,histories,candidates):
    """Return [query, candidate, feature], without receiving any target labels."""
    h=np.asarray(histories,dtype=int)
    c=np.asarray(candidates,dtype=int)
    if c.ndim==1: c=np.broadcast_to(c,(len(h),len(c)))
    q,n=c.shape
    sim=np.stack([sparse_lookup(stats.cosine,h[:,j,None],c) for j in range(5)],axis=-1)
    trans=np.stack([(sparse_lookup(stats.transition,h[:,j,None],c)+stats.prior[c]) /
                    (stats.transition_out[h[:,j,None]]+1) for j in range(5)],axis=-1)
    features=[np.log1p(stats.pop[c]),sim.mean(axis=-1),sim.max(axis=-1),sim[:,:,-1],
              np.log(trans[:,:,-1]),np.log(trans.mean(axis=-1)),(h[:,None,:]==c[:,:,None]).mean(axis=-1)]
    ha=data["artist_ids"][h]; ca=data["artist_ids"][c]
    at=np.stack([(sparse_lookup(stats.artist_transition,ha[:,j,None],ca)+stats.artist_prior[ca]) /
                 (stats.artist_out[ha[:,j,None]]+1) for j in range(5)],axis=-1)
    features += [(ha[:,-1,None]==ca).astype(float),(ha[:,None,:]==ca[:,:,None]).mean(axis=-1),
                 np.log1p(stats.artist_pop[ca]),np.log(at[:,:,-1]),np.log(at.mean(axis=-1))]
    z=data["z"]; zn=data["zn"]
    centroid=z[h].mean(axis=1)
    unit_centroid=normalize(centroid)
    last=z[h[:,-1]]
    features += [np.sum(zn[c]*zn[h[:,-1,None]],axis=-1),
                 np.sum(zn[c]*unit_centroid[:,None,:],axis=-1),
                 np.sqrt(np.sum((z[c]-last[:,None,:])**2,axis=-1)),
                 np.sqrt(np.sum((z[c]-centroid[:,None,:])**2,axis=-1))]
    diff=np.abs(z[c]-last[:,None,:])
    features += [diff[:,:,j] for j in range(len(AUDIO_COLUMNS))]
    result=np.stack(features,axis=-1).astype(np.float32)
    assert result.shape==(q,n,len(FEATURE_NAMES)) and np.isfinite(result).all()
    return result


def negative_candidates(data,queries,seed):
    """One observed target plus 10 distinct uniform unobserved candidate targets."""
    rng=np.random.default_rng(seed)
    target=queries.target.to_numpy()
    c=np.empty((len(queries),NEGATIVES+1),dtype=np.int32)
    c[:,0]=target
    for row,t in enumerate(target):
        # Draw distinct integers from N-1 and skip t, avoiding target collisions.
        sample=rng.choice(len(data["tracks"])-1,size=NEGATIVES,replace=False)
        c[row,1:]=sample+(sample>=t)
    return c


def training_examples(data):
    """Cross-fit relational features by playlist; include every training window."""
    q=data["queries"]["train"].reset_index(drop=True)
    candidates=negative_candidates(data,q,SEED+1)
    features=np.empty((len(q),NEGATIVES+1,len(FEATURE_NAMES)),dtype=np.float32)
    folds=np.array_split(data["split"]["train"],4)
    fold_manifest=[]
    for fold,held in enumerate(folds):
        fit=np.setdiff1d(data["split"]["train"],held)
        stats=TrainingStatistics(data,fit)
        assert not stats.pids.intersection(map(int,held))
        rows=np.flatnonzero(q.pid.isin(held))
        for first in range(0,len(rows),512):
            ix=rows[first:first+512]
            features[ix]=candidate_features(data,stats,q.iloc[ix][[f"h{j}" for j in range(5)]].to_numpy(),candidates[ix])
        fold_manifest.extend({"pid":int(pid),"held_fold":fold} for pid in held)
        print(f"Cross-fit fold {fold+1}/4: {len(rows):,} windows; statistics from {len(fit)} playlists",flush=True)
    pd.DataFrame(fold_manifest).to_csv(OUT / "training_feature_folds.csv",index=False)
    labels=np.zeros(candidates.shape,dtype=np.int8); labels[:,0]=1
    # Give every playlist equal total weight despite its number of overlapping windows.
    windows_per_playlist=q.groupby("pid").pid.transform("size").to_numpy()
    weights=np.repeat(1/windows_per_playlist,NEGATIVES+1)
    weights=weights/weights.mean()
    assert np.all(candidates[:,0,None]!=candidates[:,1:])
    return features.reshape(-1,len(FEATURE_NAMES)),labels.ravel(),weights


def query_features(data,stats,queries):
    catalog=np.arange(len(data["tracks"]))
    # At 100 openings, 100 x 36,203 x 26 float32 is about 377 MB.
    # Persist only small batches during evaluation, not this entire cube.
    for first in range(0,len(queries),4):
        block=queries.iloc[first:first+4]
        h=block[[f"h{j}" for j in range(5)]].to_numpy()
        yield first,candidate_features(data,stats,h,catalog)


def rank_target(scores,target,popularity):
    """1-based full-catalogue rank; ties use training support then track URI."""
    ids=np.arange(len(scores))
    score=scores[target]
    better=(scores>score)
    tie=scores==score
    better |= tie & (popularity>popularity[target])
    better |= tie & (popularity==popularity[target]) & (ids<target)
    return int(1+better.sum())


def ranking_metrics(ranks):
    r=np.asarray(ranks)
    result={"queries":len(r),"Median rank":float(np.median(r)),"Mean rank":float(np.mean(r)),
            "MRR":float(np.mean(1/r))}
    for k in [1,5,10,20,50,100]:
        result[f"Hits@{k}"]=int((r<=k).sum())
        result[f"HR@{k}"]=float(np.mean(r<=k))
        result[f"MRR@{k}"]=float(np.mean(np.where(r<=k,1/r,0)))
    return result


def choose_models(data,stats,x,y,weights):
    q=data["queries"]["validation"]
    results=[]; models={}; best={}
    # Full-catalogue validation features are built once per block per C sweep.
    for name,columns in MODEL_FEATURES.items():
        best_key=None
        for c in REGULARIZATION:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,solver="lbfgs",max_iter=500,random_state=SEED))
            start=perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error",ConvergenceWarning)
                model.fit(x[:,columns],y,logisticregression__sample_weight=weights)
            fit_seconds=perf_counter()-start
            ranks=[]
            for first,features in query_features(data,stats,q):
                scores=model.decision_function(features[:,:,columns].reshape(-1,len(columns))).reshape(features.shape[:2])
                for j,row in enumerate(q.iloc[first:first+len(features)].itertuples()):
                    ranks.append(rank_target(scores[j],row.target,stats.pop))
            metrics=ranking_metrics(ranks)
            row={"model":name,"C":c,"fit_seconds":fit_seconds,"iterations":int(model[-1].n_iter_[0]),**metrics}
            results.append(row)
            # Primary endpoint only; equal HR chooses the smaller C in the fixed grid.
            key=metrics["HR@10"]
            if best_key is None or key>best_key:
                best_key=key; models[name]=model; best[name]=row
            print(f"Validation {name}, C={c}: HR@10={metrics['HR@10']:.4f}, MRR@10={metrics['MRR@10']:.4f}",flush=True)
    table=pd.DataFrame(results)
    table.to_csv(OUT / "validation_grid.csv",index=False)
    (OUT / "frozen_selection.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
    coef=[]
    for name,model in models.items():
        for j,col in enumerate(MODEL_FEATURES[name]):
            coef.append({"model":name,"feature":FEATURE_NAMES[col],"standardized_coefficient":float(model[-1].coef_[0,j])})
    pd.DataFrame(coef).to_csv(OUT / "coefficients.csv",index=False)
    return models,best,table


def evaluate_queries(data,stats,models,queries,label,save_examples=False):
    rows=[]; examples=[]
    for first,features in query_features(data,stats,queries):
        block=queries.iloc[first:first+len(features)]
        learned={name:model.decision_function(features[:,:,MODEL_FEATURES[name]].reshape(-1,len(MODEL_FEATURES[name]))).reshape(features.shape[:2])
                 for name,model in models.items()}
        for j,row in enumerate(block.itertuples()):
            h=np.array([getattr(row,f"h{k}") for k in range(5)])
            last_artist=data["artist_ids"][h[-1]]
            scores={"Popularity":stats.pop,
                    "Last artist":(data["artist_ids"]==last_artist).astype(float),
                    "Item cosine":features[j,:,1],"Markov":features[j,:,4],
                    "Audio similarity":features[j,:,13]}
            scores.update({name:values[j] for name,values in learned.items()})
            for name,s in scores.items():
                assert len(s)==len(data["tracks"]) and np.isfinite(s).all()
                rank=rank_target(s,row.target,stats.pop)
                rows.append({"experiment":label,"pid":int(row.pid),"start":int(row.start),"model":name,
                             "target":int(row.target),"rank":rank,"track_support":int(stats.pop[row.target]),
                             "same_last_artist":bool(data["artist_ids"][row.target]==last_artist),
                             "target_in_history":bool(row.target in h)})
                if save_examples:
                    ids=np.arange(len(s)); order=np.lexsort((ids,-stats.pop,-s))[:10]
                    for k,item in enumerate(order,1):
                        examples.append({"pid":int(row.pid),"model":name,"rank":k,"item":int(item),
                                         "is_target":int(item==row.target),"track":data["tracks"].iloc[item].track_name,
                                         "artist":data["tracks"].iloc[item].artist_name})
    details=pd.DataFrame(rows)
    table=pd.DataFrame([{"model":name,**ranking_metrics(g["rank"])} for name,g in details.groupby("model",sort=False)])
    details.to_csv(OUT / (label+"_per_query.csv"),index=False)
    table.to_csv(OUT / (label+"_metrics.csv"),index=False)
    if save_examples: pd.DataFrame(examples).to_csv(OUT / "test_top10.csv",index=False)
    return details,table


def paired_bootstrap(before,after):
    a=np.asarray(before); b=np.asarray(after)
    assert a.shape==b.shape
    rng=np.random.default_rng(2026)
    ix=rng.integers(0,len(a),size=(10000,len(a)))
    delta=(b<=10).astype(float)-(a<=10).astype(float)
    lo,hi=np.quantile(delta[ix].mean(axis=1),[.025,.975])
    gained=int(((b<=10)&(a>10)).sum()); lost=int(((a<=10)&(b>10)).sum())
    # Exact paired binary test; only discordant query outcomes count.
    p=binomtest(gained,gained+lost,.5).pvalue if gained+lost else 1.
    return {"delta_hr10":float(delta.mean()),"ci_low":float(lo),"ci_high":float(hi),
            "gained_hits":gained,"lost_hits":lost,"exact_mcnemar_p":float(p),
            "delta_mrr10":float(np.mean(np.where(b<=10,1/b,0)-np.where(a<=10,1/a,0))),
            "median_paired_rank_change":float(np.median(b-a)),"bootstrap_draws":10000}


def summarize_and_plot(data,stats,details,table,models,selection):
    subgroup=[]
    for name,g in details.groupby("model",sort=False):
        for support,sub in g.groupby(g.track_support.gt(0)):
            subgroup.append({"model":name,"support":"positive" if support else "zero",**ranking_metrics(sub["rank"])})
    subgroup=pd.DataFrame(subgroup)
    subgroup.to_csv(OUT / "test_support_metrics.csv",index=False)
    before=details[details.model=="History + artist"].set_index("pid").sort_index()
    after=details[details.model=="History + artist + audio"].set_index("pid").sort_index()
    assert before.index.equals(after.index)
    primary=paired_bootstrap(before["rank"],after["rank"])
    pairs=[]
    for pid in before.index:
        pairs.append({"pid":int(pid),"rank_without_audio":int(before.loc[pid,"rank"]),
                      "rank_with_audio":int(after.loc[pid,"rank"]),"track_support":int(before.loc[pid,"track_support"])})
    pd.DataFrame(pairs).to_csv(OUT / "audio_paired_ranks.csv",index=False)
    n=len(data["tracks"])
    random={"model":"Uniform random (expectation)","queries":len(before),"HR@10":10/n,
            "MRR@10":float(np.sum(1/np.arange(1,11))/n),"Median rank":(n+1)/2}
    # First test opening chosen by sorted pid, never by success.
    example_pid=int(data["queries"]["test"].pid.min())
    summary={"primary":primary,"frozen_selection":selection,"random_baseline":random,
             "test_queries":len(before),"zero_support_targets":int((before.track_support==0).sum()),
             "test_same_last_artist_fraction":float(before.same_last_artist.mean()),
             "test_repeat_target_fraction":float(before.target_in_history.mean()),"example_pid":example_pid,
             "train_windows":len(data["queries"]["train"]),"negatives_per_window":NEGATIVES,
             "training_candidate_examples":len(data["queries"]["train"])*(NEGATIVES+1),
             "feature_names":FEATURE_NAMES,"environment":{"python":platform.python_version(),"numpy":np.__version__,
                "pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
             "dataset_sha256":data["summary"]["dataset_sha256"]}
    (OUT / "experiment_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    names=table.model.to_list()
    colors=["#a3adb4"]*5+["#8bacbf","#297b69","#d18e3b"]
    fig,ax=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    ax[0].barh(names,table["HR@10"],color=colors)
    ax[0].invert_yaxis();ax[0].set(xlabel="Exact sixth-track Hit Rate@10",title="What appears in a short list?")
    for i,value in enumerate(table["HR@10"]): ax[0].text(value+.003,i,f"{int(round(value*len(before)))}/{len(before)}",va="center",fontsize=8)
    ax[0].set_xlim(0,max(table["HR@10"])+.07)
    ax[1].barh(names,table["Median rank"],color=colors)
    ax[1].invert_yaxis(); ax[1].set(xlabel="Median target rank (log; lower is better)",xscale="log",title="What moves below the cutoff?")
    fig.savefig(OUT / "retrieval_comparison.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,3.6),layout="constrained")
    for name in MODEL_FEATURES:
        ranks=details.loc[details.model==name,"rank"].to_numpy()
        ks=np.array([1,5,10,20,50,100,500,1000])
        ax[0].plot(ks,[(ranks<=k).mean() for k in ks],marker="o",label=name)
    ax[0].set(xscale="log",xlabel="Recommendation cutoff K (log)",ylabel="Hit Rate@K",title="The useful cutoff changes the comparison")
    ax[0].legend(fontsize=8)
    lim=n+1
    for support,color in [(False,"#3979ad"),(True,"#d18e3b")]:
        mask=before.track_support.gt(0).to_numpy()==support
        ax[1].scatter(before["rank"].to_numpy()[mask],after["rank"].to_numpy()[mask],s=20,alpha=.7,
                      label="Positive support" if support else "Zero support",color=color)
    ax[1].plot([1,lim],[1,lim],color="gray",linestyle="--")
    ax[1].set(xscale="log",yscale="log",xlabel="Rank without audio",ylabel="Rank with audio",title="Below diagonal: audio improves rank")
    ax[1].legend(fontsize=8)
    fig.savefig(OUT / "audio_ablation.png",dpi=170);plt.close(fig)
    return summary,subgroup


def sensitivity_checks(data,stats,models):
    openings=data["queries"]["test"].copy()
    history=[f"h{j}" for j in range(5)]
    reversed_q=openings.copy()
    reversed_q[history]=openings[history].to_numpy()[:,::-1]
    _,reverse_table=evaluate_queries(data,stats,models,reversed_q,"sensitivity_reverse")
    # An exact invariance diagnostic: internal first-four order is not represented.
    permuted=openings.copy()
    permuted[history]=openings[history].to_numpy()[:,[3,1,0,2,4]]
    _,permuted_table=evaluate_queries(data,stats,models,permuted,"sensitivity_first4")
    later=queries_from_sequences(data["sequences"],data["split"]["test"],start=5)
    _,later_table=evaluate_queries(data,stats,models,later,"sensitivity_later")
    matched=openings[openings.pid.isin(later.pid)].reset_index(drop=True)
    _,matched_table=evaluate_queries(data,stats,models,matched,"sensitivity_matched_openings")
    parts=[]
    for label,t in [("Reversed five",reverse_table),("First four permuted; last fixed",permuted_table),
                    ("Positions 5-9 -> 10",later_table),("Opening, matched later playlists",matched_table)]:
        parts.append(t.assign(setting=label))
    table=pd.concat(parts,ignore_index=True)
    table.to_csv(OUT / "sensitivity_metrics.csv",index=False)
    return table


def run_experiment(data=None):
    if data is None: data=prepare_data()
    stats=TrainingStatistics(data,data["split"]["train"])
    assert not stats.pids.intersection(map(int,data["split"]["validation"]))
    assert not stats.pids.intersection(map(int,data["split"]["test"]))
    x,y,weights=training_examples(data)
    print(f"Fitting {len(y):,} sampled candidates with {x.shape[1]} features",flush=True)
    models,selection,validation=choose_models(data,stats,x,y,weights)
    # All model fitting and selection is finished before test labels are evaluated.
    details,table=evaluate_queries(data,stats,models,data["queries"]["test"],"test",save_examples=True)
    summary,subgroup=summarize_and_plot(data,stats,details,table,models,selection)
    sensitivity=sensitivity_checks(data,stats,models)
    print(table[["model","HR@10","MRR@10","Median rank"]].to_string(index=False),flush=True)
    print("Primary paired comparison:",json.dumps(summary["primary"]),flush=True)
    return {"data":data,"stats":stats,"models":models,"selection":selection,"validation":validation,
            "details":details,"table":table,"summary":summary,"subgroup":subgroup,"sensitivity":sensitivity}


if __name__=="__main__":
    with threadpool_limits(limits=4):
        run_experiment()
