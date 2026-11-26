"""Build and execute the documented notebook, then export a standalone HTML file."""
from pathlib import Path
import ast
import os
import sys
import textwrap

ROOT = Path(__file__).resolve().parent
for name, folder in [("JUPYTER_RUNTIME_DIR","jupyter"),("JUPYTER_CONFIG_DIR","jupyter_config"),
                     ("IPYTHONDIR","ipython"),("MPLCONFIGDIR","matplotlib")]:
    os.environ[name] = str(ROOT / "tmp" / folder)
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

source = (ROOT / "experiment.py").read_text(encoding="utf-8")
tree = ast.parse(source)
definitions = {n.name: ast.get_source_segment(source,n) for n in tree.body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
cells=[]


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))


def functions(*names):
    code("\n\n\n".join(definitions[n] for n in names))


md(r"""
# Spotify playlist completion: Assignment 2, parts 2-4

**Exploratory analysis · Modeling · Evaluation**

This notebook completes parts **2, 3, and 4 of the attached assignment PDF**.
The lecture overview numbers the topics differently; the PDF's numbering is used here.
The task is to recover withheld tracks from an existing playlist. The inputs are its
observed tracks, the fitting playlists' interactions, and optional static audio metadata;
the output is a ranked list of tracks absent from the seed playlist.

**Comparison:** a training-popularity baseline plus three distinct models:
playlist-neighborhood collaborative filtering, PureSVD latent factors, and an audio-centroid
content model. Model and feature selection use validation NDCG@20. The test set is used
only after selection. All outputs below are executed results, not illustrative numbers.

**Reading guide:** Part 2 explains the data and preprocessing; Part 3 describes the
models and implementation; Part 4 gives the evaluation protocol, comparisons, uncertainty,
and limitations. Code is included so the HTML can be reviewed without executing it.

**Scope:** this is a partial assignment deliverable covering the requested sections.
It does not include a recorded presentation or a peer-grading submission.
""")

md("""
## 2. Exploratory analysis, data collection, and preprocessing

### 2.1 Data context and provenance

The supplied file, `data/songs620.csv`, is the dataset used throughout. Each row associates
a Spotify track with a playlist ID (`pid`) and a playlist position (`pos`). It also includes
artist/album identifiers, names, duration, and Spotify-style audio descriptors.
The filename does **not** imply that the file contains only 620 songs.

Its playlist schema and IDs 0-999 are consistent with an enriched subset of Spotify's
Million Playlist Dataset (MPD). **That origin is an inference:** the repository has no
collection script, source URL, license record, sampling description, or enrichment date,
so this notebook cannot authenticate the exact source or how the audio features were fetched.
The observed data are therefore described as the **supplied Spotify playlist CSV**.
No new Spotify API collection is performed, and URL columns are never fetched.

For context, Spotify describes the original MPD as a resource for automatic playlist
continuation, containing one million playlists and over two million tracks [R1]. That
description concerns the original dataset; this CSV is much smaller. The data contain
no listener IDs, ratings, skip logs, playlist titles, or observation timestamps. A playlist
is the recommendation context, **not a verified individual listener**. `pos` is an ordering
field rather than a timestamp, so this study makes no temporal forecasting claim.
""")

prefix = source[:source.index("def load_data")]
prefix = prefix[prefix.index("from __future__"):]
prefix = prefix.replace('Path(__file__).resolve().parent','Path.cwd()')
code(prefix + "\nfrom IPython.display import display, Image, Markdown\n"
     "_thread_limit = threadpool_limits(limits=4)\n"
     "assert (ROOT / 'data/songs620.csv').exists(), 'Run from the repository root.'\n")

md("""
### 2.2 Clean interactions before splitting

An interaction is binary playlist membership: a track is either present or absent.
Repeated occurrences of a track in one playlist are collapsed **before** splitting, so
the same membership cannot appear in both a seed and its held-out targets. Distinct
playlists sharing a track are retained: that overlap is the collaborative signal.

Track metadata are checked for inconsistencies across repeated occurrences, then one
consistent record per track is used. The index column `Unnamed: 0`, redundant track IDs,
URLs, names, album fields, and `pos` do not enter the predictive models. Names are used
only for readable examples; artist IDs are used for the diversity diagnostic.
""")
functions("load_data","split_edges","audio_features","run_eda")

# Reuse the exact experiment body in notebook cells; it is not an import-only wrapper.
body = definitions["run"]
body = textwrap.dedent(body[body.index("\n")+1:])
eda_body,rest = body.split('configs = [("Popularity", {})]',1)
validation_body = 'configs = [("Popularity", {})]' + rest
validation_body,test_body = validation_body.split("# Test data are first used here, after the full selection is frozen.",1)
test_body = test_body.rsplit("return experiment_summary",1)[0]
code(eda_body)
code("""
overview = pd.DataFrame({
    'Quantity': ['Raw rows', 'Repeated playlist-track rows removed', 'Unique memberships',
                 'Playlists', 'Tracks', 'Artists', 'Singleton tracks', 'Interaction density'],
    'Value': [summary['raw_rows'], summary['duplicate_playlist_track_rows'],
              summary['unique_interactions'], summary['playlists'], summary['tracks'],
              summary['artists'], summary['singleton_tracks'], f"{summary['density']:.4%}"]})
display(overview)
display(pd.DataFrame(summary['playlist_size'], index=['Unique playlist size']).T.round(2))
display(pd.read_csv(OUT / 'audio_summary.csv', index_col=0).round(3))
display(Image(filename=str(OUT / 'eda.png')))
""")
md("""
### 2.3 What the data show, and why it matters

There are **66,263 raw rows**, reduced to **65,409 unique playlist-track memberships**
after removing **854 repeated memberships**. The file contains **1,000 playlists**,
**36,203 tracks**, and **9,963 artists**. Only **0.181%** of the playlist-track matrix
is observed. Playlist lengths range from 5 to 246 unique tracks, with a median of 49.

The track distribution has a substantial long tail: **26,901 tracks (74.3%) appear in
only one playlist**, while the most frequent 1% of tracks account for 10.5% of unique
memberships. Popularity is therefore a relevant baseline, but many test targets will
lack collaborative evidence. This motivates comparing a content model with two
collaborative models, and reporting warm-track and cold-track performance separately.

No supplied column has missing values, and the numerical model inputs are finite.
Repeated metadata records agree for all modeled features and artist IDs. A defensive
median imputer remains in the content pipeline; it is fitted only on tracks observed
in fitting interactions. Duration is log-transformed and continuous features are
standardized on those same fitting tracks. Held-out track metadata can be transformed
because the catalogue is assumed known; their interaction labels are never used to fit
preprocessing. Zero tempo or energy values are retained as supplied rather than silently
interpreted as missing. Their upstream meaning cannot be checked from this CSV.

Energy and loudness correlate strongly at the unique-track level (Pearson r = 0.788).
This suggests redundant descriptors and motivates testing a compact feature set against
a fuller one. Correlation alone does not establish predictive usefulness.
""")
code("""
display(pd.read_csv(OUT / 'missingness.csv'))
print('Tracks with conflicting modeled metadata:', summary['metadata_conflicting_tracks'])
print('Non-finite numerical values:', sum(summary['invalid_numeric_values'].values()))
print('Training within-playlist mean core-audio cosine:',
      round(summary['training_audio_cosine_within_mean'], 4))
print('Random catalogue-pair mean core-audio cosine:',
      round(summary['catalogue_audio_cosine_random_mean'], 4))
display(Image(filename=str(OUT / 'audio_coherence.png')))
""")
md("""
The training-only within-playlist pair sample has mean audio cosine similarity **0.2360**,
versus **0.0135** for random catalogue pairs. This is descriptive evidence that audio
features capture some playlist coherence. Pairs share playlists and tracks, so they are
not treated as independent observations for a significance test. Similar audio also
does not guarantee recovery of an exact withheld track from 36,203 candidates.

### 2.4 Split and information boundary

Within each playlist, unique tracks are shuffled using seed 42. Two disjoint sets of
`floor(0.2 * playlist_length)` tracks are assigned to validation and test; the remainder
forms training. All playlists have at least five tracks, so every partition has at
least one interaction per playlist. The resulting split has **40,013 training**,
**12,698 validation**, and **12,698 test** memberships.

During selection, models fit only the training matrix and recommend against validation
targets. After all configurations are frozen, each model is refitted on training plus
validation (**52,711 memberships**) and recommends against the untouched test targets.
This uses the validation tracks as additional seeds at final evaluation, for every model.
Validation and test scores consequently have different seed lengths and are not a
train-test generalization-gap estimate.

This is a **random missing-track completion** experiment on known playlists and a known
static catalogue. It does not measure new-playlist performance, future listening,
track ordering, or chronological next-song prediction. Tracks may occur in several
playlists; each individual membership is assigned exactly once. The complete mapping
is saved in `output/split_manifest.csv` with separate item and playlist vocabularies.
""")
code("""
display(pd.DataFrame({'split': list(matrices),
                      'memberships': [m.nnz for m in matrices.values()]}))
assert train.multiply(validation).nnz == 0
assert train.multiply(test).nnz == 0
assert validation.multiply(test).nnz == 0
assert (train + validation + test).nnz == len(edges)
print('Verified: disjoint memberships, complete accounting, and nonempty playlist partitions.')
""")

md(r"""
## 3. Modeling

### 3.1 Inputs, outputs, and course connections

Let $B_{pi}=1$ when fitting playlist $p$ contains track $i$, and zero otherwise.
For each playlist the model outputs a score $s(p,i)$ for every track in the catalogue.
Observed seed tracks are removed before taking the highest scores. Unobserved
memberships are **unknown**, rather than explicit dislikes.

The lecture `recommendation.pdf` [L1] introduces cosine similarity, neighborhood
collaborative filtering, low-dimensional latent factors, implicit feedback, and ranking
metrics. Those ideas motivate the first two learned approaches and the evaluation.
`regression.pdf` [L2] motivates categorical encodings, feature transforms, control of
model complexity, and a distinct validation set. The implementation uses those ideas
for audio features and model selection. The latent-factor model below is a truncated
SVD adaptation, not a claim to reproduce the lecture's rating-loss optimizer exactly.

### 3.2 Baseline and three models

| Method | Score or objective | Role and tradeoff |
|---|---|---|
| Popularity baseline | $s(p,i)=\sum_q B_{qi}$ | Ranks tracks by fitting playlist count. Cheap and robust, but has no playlist-specific preferences and favors popular tracks. |
| Playlist-kNN collaborative filtering | $c(p,q)=B_pB_q^T/(\|B_p\|\|B_q\|)$; $s(p,i)=\sum_{q\in N_k(p),q\ne p}c(p,q)^a B_{qi}$ | Uses shared tracks to find related playlists. Interpretable, but sparse overlap and cold items are difficult. The diagonal similarity is set to zero. |
| PureSVD latent factors | Approximate $\min_{\mathrm{rank}(\hat B)\le d}\|B-\hat B\|_F^2$ using randomized truncated SVD; $\hat B=(BV_d)V_d^T$ | Low-rank representations share information across playlists. Rank limits complexity. The zero-filled reconstruction objective is only a surrogate for ranking and gives unknown entries a zero target. |
| Audio centroid content model | Normalize item feature vectors $z_i$; $h_p=\mathrm{normalize}(\sum_{i:B_{pi}=1}z_i)$; $s(p,i)=h_p^Tz_i$ | Fits scaling from known tracks and builds a playlist profile. It can score catalogue tracks without fitting interactions. A single centroid may blur mixed-genre playlists and cannot capture all cultural or semantic preferences. |

PureSVD uses `TruncatedSVD` with seven power iterations and fixed randomness. The
returned transformed matrix is $BV_d$, so multiplying by `components_` produces the
rank-$d$ reconstruction. There is no stochastic-gradient learning rate or early
stopping to tune. Its chosen rank is selected only with validation NDCG@20.

The centroid is a content model fitted through preprocessing and profile construction;
it does not learn a supervised neural network. The three models represent neighborhood,
latent-factor, and content-based approaches, in addition to the baseline.
""")
md("""
### 3.3 Feature representations and hyperparameter search

**Collaborative features:** a sparse binary playlist-track matrix. Frequencies are
computed after deduplication, not from repeated positions. Neither collaborative model
uses audio, names, artist IDs, or test interactions.

**Core audio representation:** danceability, energy, loudness, valence, and tempo.
**Full representation:** those five plus speechiness, acousticness, instrumentalness,
liveness, and log-duration, followed by fixed one-hot encodings of key, mode, and time
signature. The encodings avoid imposing a numerical ordering on categories; they do
not model harmonic distance. Reserved categories make the full vector 35-dimensional.
Continuous columns are standardized; category indicator columns remain 0/1. Each
resulting item vector is L2-normalized. These choices impose equal coordinate weights
rather than learning the ideal importance of each descriptor.

The predefined grid contains **12 configurations**: popularity (1), kNN with
neighbors in {20, 50, 100} and similarity power in {1, 2} (6), SVD with rank in
{16, 32, 64} (3), and core/full audio (2). Each model family selects its highest
validation NDCG@20 configuration. A stable first-configuration rule breaks any exact
ties. No grid is expanded after inspecting test scores.
""")
functions("Ranker","top_items","recommend")
md("""
### 3.4 Architectural choices, efficiency, and overfitting

The binary matrix uses SciPy CSR storage, with one stored value per fitting
membership. The kNN implementation builds the playlist-by-playlist similarity matrix,
then retains only the selected neighbors. With 1,000 playlists this dense intermediate
is approximately 4 MB in float32. It is practical here, but its quadratic playlist
memory would be inappropriate at MPD scale; approximate neighbor retrieval or blocked
similarity search would be needed there.

SVD stores approximately $d(P+I)$ factor entries, and content scoring stores $I F$
item-feature entries. Ranking scores are generated in batches of 32 playlists, avoiding
a full playlist-by-catalogue score matrix. A batch of 32 by 36,203 float32 scores is
about 4.6 MB. CSR interactions and batched multiplication keep this dataset small
enough for CPU execution. Reported times cover fitting and top-50 scoring/ranking,
not CSV parsing, EDA, metric calculation, or bootstrap uncertainty.

Top-k selection uses partitioning, with explicit tie handling: descending model score,
then descending fitting popularity, then stable track ID. Thus every model uses the
same deterministic fallback for tied or unsupported predictions. Content scores can
be negative after centering; their ordering remains valid. Seed tracks are always
masked, and a playlist is never its own kNN neighbor.

The rank constraint regularizes SVD; k and the similarity power control neighborhood
locality. The audio model has no supervised coefficient fitting, but representation
selection still risks validation overfitting. A fixed grid and a held-out test set
limit that risk. Training reconstruction is not reported as recommendation quality,
since rewarding recovery of already-seen tracks would be misleading.

Rating regression is not directly supported because no ratings exist. Ordinary
accuracy on all playlist-track pairs would be dominated by missing entries. A
sequence transformer would require a different ordered-task design and substantially
more training context. These models could be studied later, but are not necessary
to compare the course's similarity and latent-factor approaches here.
""")

md(r"""
## 4. Evaluation

### 4.1 Metrics and candidate set

Every model ranks the **entire 36,203-track catalogue**, removing only tracks in that
playlist's fitting seed. No sampled negative set or minimum-popularity filter is used.
Validation targets are not removed during validation; test targets are not removed
during test evaluation. Catalogue metadata are assumed available even for tracks with
zero fitting interactions.

The primary metric is **macro NDCG@20**, which rewards retrieving relevant tracks near
the top of a short recommendation list. For held-out set $T_p$, and relevance
$r_{pj}=1$ if the track at rank $j$ is in $T_p$:

$$\mathrm{DCG@K}(p)=\sum_{j=1}^{K}\frac{r_{pj}}{\log_2(j+1)},\qquad
\mathrm{NDCG@K}(p)=\frac{\mathrm{DCG@K}(p)}
{\sum_{j=1}^{\min(K,|T_p|)}1/\log_2(j+1)}.$$

**Recall@K** is the number of retrieved targets divided by $|T_p|$;
**Precision@K** divides by K; **HitRate@K** is one if at least one target is retrieved.
Each metric is averaged equally over playlists, so long playlists do not dominate.
We report K = 10, 20, and 50, with K = 20 fixed before selection.

**Coverage@20** is the number of distinct tracks recommended across all playlists
divided by the catalogue size. **Artist diversity@20** is the mean number of distinct
artists per list divided by 20. **Novelty@20** is the mean
$\log_2((P+1)/(n_i+1))$ of recommended tracks, using fitting playlist count $n_i$.
Higher novelty therefore means lower fitting popularity; it is not evidence that a
listener enjoys the recommendations.

NDCG and recall match the recovery task more directly than MSE, which would compare
scores without calibrated rating labels, or classification accuracy, which treats
nearly all unknown pairs as negatives. Ranking relevance is still incomplete:
unobserved tracks could be enjoyable, so this is exact-membership recovery rather
than a direct measure of satisfaction. The metrics are not numerically comparable
to the official MPD challenge's differently configured evaluation [R2].
""")
functions("evaluate","paired_uncertainty")
md("""
### 4.2 Validation selection

The following code runs the declared grid, computes validation metrics, and freezes
one configuration per family. It chooses the overall model by validation NDCG@20,
not by the test leaderboard. The full grid is retained to show the feature ablation
and sensitivity to model complexity.
""")
code(validation_body)
code("""
display(val_table[['model','parameters','NDCG@20','Recall@20','Precision@20']].round(5))
print('Frozen validation-selected model:', winner)
display(pd.DataFrame(selected).T[['parameters','NDCG@20']])
""")
md("""
Validation selects **50 neighbors with power 1**, **SVD rank 64**, and the **full audio
representation**. Audio validation NDCG@20 increases from **0.00442** for the core
representation to **0.00932** for the full one. Because several descriptors and
categorical features change together, this comparison does not isolate which added
feature causes the gain. The fixed grid ends at rank 64, so this study does not claim
that rank 64 is globally optimal. Increasing k from 50 to 100 does not improve the
validation score; a broader neighborhood is not automatically better.
""")
md("""
### 4.3 Final refit, test evaluation, and uncertainty

Each frozen configuration is refitted on train plus validation. The next code is the
first experiment stage that uses test labels. It saves per-playlist results and
top-50 recommendations so the aggregate values can be audited.

For the primary metric, **2,000 paired bootstrap samples of playlists** produce a
95% percentile interval for each model's difference from popularity. We also use a
**10,000-draw two-sided paired sign-flip test** with the +1 Monte Carlo correction.
Holm adjustment covers the three model-versus-baseline tests. Mean-score intervals
on the plot also resample playlists. These intervals are conditional on this split
and fitted models; shared tracks and overlapping playlist tastes mean playlist
independence is an approximation. They do not quantify uncertainty from recollecting
the dataset, changing the split, or refitting on bootstrap datasets.
""")
code(test_body)
code("""
display(result_table[['model','NDCG@20','Recall@20','Precision@20','HitRate@20',
                      'Coverage@20','fit_seconds','rank_seconds']].round(5))
display(Image(filename=str(OUT / 'model_comparison.png')))
display(pd.read_csv(OUT / 'paired_comparisons.csv').round(6))
display(result_table[['model']+[f'{m}@{k}' for k in [10,20,50]
                              for m in ['NDCG','Recall','Precision','HitRate']]].round(5))
""")
md("""
### 4.4 Main findings

**Playlist-kNN is the validation-selected model and has the strongest observed test
performance.** It reaches **NDCG@20 = 0.07098**, **Recall@20 = 0.07222**, and
**HitRate@20 = 0.434**. Popularity reaches **0.01081**, **0.01244**, and **0.100**,
respectively. The paired NDCG improvement is **0.06016**, with a 95% interval of
**[0.05370, 0.06727]** and Holm-adjusted p = **0.00030**. Thus shared-track
neighborhoods add a clear signal over overall fitting popularity on this split.
The sign-flip p-value is limited by the finite number of randomization draws.

**PureSVD also clearly improves over popularity**, reaching NDCG@20 = **0.05891**
and Recall@20 = **0.06124**. Its paired NDCG difference is **0.04810**
(95% interval **[0.04239, 0.05392]**, Holm-adjusted p = **0.00030**).
The kNN score is higher than SVD's, although the formal tests shown here compare
each model with popularity and do not test the kNN-SVD difference. Local overlaps
may preserve playlist-specific associations that a 64-dimensional reconstruction
smooths away; this is an interpretation, not an experimentally isolated cause.

**The audio-centroid model does not beat popularity overall.** Its test NDCG@20 is
**0.00885** and Recall@20 is **0.01175**. Its NDCG difference from popularity is
**-0.00196**, with interval **[-0.00562, 0.00219]** and adjusted p = **0.32437**.
The interval includes zero. Audio coherence in EDA therefore does not translate into
reliable exact-track recovery with this simple model. Artist identity, language,
semantic theme, and multi-modal playlist structure may matter beyond a centroid of
low-dimensional descriptors. Those explanations would need additional ablations.

Absolute recall remains modest: even kNN recovers about 7.2% of a playlist's held-out
tracks in its top 20, on average. This is a difficult sparse-catalogue task, and a
large improvement over a weak baseline is not equivalent to a production-ready system.
""")
code("""
cold_share = experiment_summary['cold_test_interactions'] / experiment_summary['test_interactions']
print(f'Cold target interactions: {experiment_summary["cold_test_interactions"]:,} / '
      f'{experiment_summary["test_interactions"]:,} = {cold_share:.2%}')
display(result_table[['model','WarmRecall@20','ColdRecall@20','Coverage@20',
                      'ArtistDiversity@20','Novelty@20']].round(5))
display(pd.read_csv(OUT / 'length_subgroups.csv').round(5))
""")
md("""
### 4.5 Cold tracks, coverage, and playlist length

**5,623 of 12,698 test interactions (44.3%)** concern tracks with zero interactions
in the final fitting matrix. This is a cold-interaction condition within a known
catalogue, not absence of metadata. Such tracks have no positive popularity or
neighborhood evidence, and SVD has no observed column signal. All three collaborative/
popularity approaches have zero cold-target Recall@20 in this run. The audio model
achieves **0.01155**. Warm Recall@20 is **0.11779** for kNN, **0.09147** for SVD,
**0.01701** for popularity, and **0.00522** for audio.

Warm and cold recalls are macro averages over the playlists with at least one target
of that type; they need not combine to the overall recall using the global cold
interaction fraction. Cold status is defined solely by the final fitting matrix.

The content model trades recovery for reach: it recommends **22.69%** of catalogue
tracks across its top-20 lists, compared with **10.24%** for kNN, **6.31%** for SVD,
and **0.091%** for popularity (33 distinct tracks). Audio also has higher artist
diversity and lower-popularity recommendations. These metrics show a different
behavior, not automatically better recommendations. A future hybrid could use
content for cold candidates and collaborative scores for warm candidates, but no
unrun hybrid result is claimed here.

For kNN, NDCG@20 is **0.0524**, **0.0665**, and **0.0869** for playlists with 1-20,
21-50, and 51+ fitting seeds. These groups contain 264, 336, and 400 playlists.
More seeds provide more context, but group differences also reflect playlist
content and target counts; they do not establish a causal benefit of adding seeds.
""")
md("""
### 4.6 A fixed example and practical recommendation

The example below is playlist index 0, chosen in advance rather than selected for a
high score. `test_hit` means exact held-out membership, not a human relevance judgment.
Inspecting titles and artists helps explain the output, but does not replace the
full-catalogue comparison.
""")
code("""
display(pd.read_csv(OUT / 'example_seeds.csv').head(12))
display(pd.read_csv(OUT / 'example_recommendations.csv'))
""")
md("""
For this dataset and completion setting, **use playlist-kNN with 50 neighbors and
cosine weights** as the primary method. It is simple, fast at this scale, and gives
the best validation and observed test performance. Keep popularity as a fallback.
PureSVD is a compact alternative; content deserves further study for tracks without
interaction evidence, despite its weak aggregate recovery.

The main limitations are the unverified extraction/enrichment history, a small
playlist sample, many singleton tracks, one random split, a known-playlist evaluation,
and incomplete relevance labels. There are no listener IDs or timestamps with which
to test personalization across people or chronological generalization. No listening
study establishes user satisfaction, diversity preference, or fairness. A larger
sample, repeated predefined splits, a new-playlist holdout, and a carefully validated
hybrid would provide stronger evidence. Test results here are final reporting, not
feedback for another round of tuning.
""")

md("""
### References and relationship to the supplied slides

**[L1] Julian McAuley, course lecture, _Recommender Systems_ (`recommendation.pdf`).**
[Supplied Google Drive slides](https://drive.google.com/file/d/1VAt1C5Za5JCBXUkeRbqkH8jajgTdfmnl/view).
Relevant topics: similarity-based recommenders, cosine similarity, latent factors,
cold start, and precision/recall at K.

**[L2] Julian McAuley, course lecture, _Supervised learning - Regression_ (`regression.pdf`).**
[Supplied Google Drive slides](https://drive.google.com/file/d/19xh-0pTHUjTMNLFCGQVnPPgivAhbnpFX/view).
Relevant topics: categorical features, feature transforms, overfitting, regularization,
and train/validation/test separation.

**[L3] Course lecture, _Assignment 2_ (`assignment2_fa26.pdf`).**
[Supplied overview](https://drive.google.com/file/d/1CwsbNGf98gcQRfw2FPCpoRAoJoqlTbcQ/view).
Used for guidance on feature comparisons, baselines, scaling, and model selection.
Section numbering in this notebook follows the separately attached assignment PDF.

**[R1] Ching-Wei Chen (2020), _The Million Playlist Dataset... Remastered_.**
[Spotify Research](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered).
Primary context for the original MPD and its playlist-continuation purpose;
it does not authenticate the local CSV's derivation.

**[R2] Hamed Zamani, Markus Schedl, Paul Lamere, and Ching-Wei Chen (2019),
_An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic
Music Playlist Continuation_.** [Paper](https://arxiv.org/abs/1810.01520).
The challenge studies continuation from seed tracks and metadata. Its evaluation
protocol differs from the present small-sample top-20 experiment, so its scores
are not used as comparable baselines.

**[R3] Irene Teinemaa, Niek Tax, and Carlos Bentes (2018), _Automatic Playlist
Continuation through a Composition of Collaborative Filters_.**
[Paper](https://arxiv.org/abs/1808.04288).
The authors combine collaborative filters for different playlist aspects. This
motivates testing interaction-based approaches; the present kNN/SVD comparison
does not reproduce their ensemble or optimization procedure.

**[R4] Andres Ferraro, Dmitry Bogdanov, Jisang Yoon, KwangSeob Kim, and Xavier Serra
(2018; arXiv 2019), _Automatic playlist continuation using a hybrid recommender
system combining features from text and audio_.**
[Paper](https://arxiv.org/abs/1901.00450).
Their system combines an audio/title-aware factorization model with a co-occurrence
model using ranking fusion. The simple audio-only centroid here is a much more
limited comparison; its negative aggregate result does not refute that hybrid approach.
""")
md("""
### Reproducibility and verification

Run from the repository root with the dependencies in `requirements.txt` installed.
The notebook contains the preprocessing, model, and evaluation implementations;
`experiment.py` runs the same experiment from the command line. The dataset checksum,
package versions, split manifest, selected configurations, per-playlist metrics,
recommendations, and aggregate CSV tables are saved under `output/`.
`workbook.html` embeds tables and figures for review without a kernel or data download.

The notebook checks split disjointness, complete interaction accounting, nonempty
partitions, unique recommendations, and exclusion of seeds. The separate
`verify_experiment.py` checks ranking ties, exclusion of self-neighbors, hand-calculated
metrics, and the saved recommendation files. Times are measured on the current CPU
with numerical thread pools limited to four; they are not a hardware-independent benchmark.
""")
code("""
display(pd.Series(experiment_summary['environment'], name='version'))
print('Dataset SHA-256:', experiment_summary['dataset_sha256'])
print('Output tables:', ', '.join(p.name for p in sorted(OUT.glob('*.csv'))))
""")

notebook=nbformat.v4.new_notebook(cells=cells,metadata={
    "kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
    "language_info":{"name":"python","version":"3.14"}})
path=ROOT / "assignment_2.ipynb"
nbformat.write(notebook,path)
print(f'Created {len(cells)} cells. Executing notebook...',flush=True)
client=NotebookClient(notebook,timeout=600,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}})
client.execute()
nbformat.write(notebook,path)
exporter=HTMLExporter(template_name="lab")
exporter.exclude_input_prompt=True
exporter.exclude_output_prompt=True
html,_=exporter.from_notebook_node(notebook)
# Match the assignment's required doctype exactly and remove remote math rendering.
# MathJax is optional; all accompanying prose explains the formulas offline.
html=html.replace('<!DOCTYPE html>', '<!DOCTYPE html>',1)
if not html.startswith('<!DOCTYPE html>'):
    html='<!DOCTYPE html>\n'+html[html.lower().index('<html'):]
css='''<style>
body{background:#f3f5f6!important;color:#17252c!important;}
main{max-width:1120px;margin:24px auto;background:white;padding:22px 35px;box-shadow:0 3px 22px #17252c12;}
h1{color:#165746!important;font-size:2.1rem!important;} h2{color:#165746!important;border-bottom:2px solid #d4e6df;padding-bottom:10px!important;margin-top:40px!important;}
h3{color:#245367!important;margin-top:28px!important;}
.jp-RenderedHTMLCommon p,.jp-RenderedHTMLCommon li{line-height:1.65!important;}
.jp-RenderedHTMLCommon table{font-size:12px!important;display:block;overflow-x:auto;max-width:100%;}
.jp-InputArea{max-height:460px;overflow:auto;border-radius:5px;}
.jp-OutputArea-output{overflow-x:auto;max-width:100%;}
@media print{body{background:white!important;}main{box-shadow:none;margin:0;padding:0;}.jp-InputArea{max-height:none;overflow:visible;}h2,h3{break-after:avoid;}img{break-inside:avoid;}}
</style>'''
html=html.replace('</head>',css+'</head>')
(ROOT / 'workbook.html').write_text(html,encoding='utf-8')
print('Executed notebook and exported workbook.html',flush=True)
