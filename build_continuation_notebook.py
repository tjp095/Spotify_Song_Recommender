"""Assemble and execute the five-section notebook. Deliberately does not export HTML."""
from pathlib import Path
import ast
import os
import textwrap
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
for name,folder in [('JUPYTER_RUNTIME_DIR','jupyter'),('JUPYTER_CONFIG_DIR','jupyter_config'),
                    ('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
    os.environ[name]=str(ROOT/'tmp'/folder)
    Path(os.environ[name]).mkdir(parents=True,exist_ok=True)
source=(ROOT/'continuation_experiment.py').read_text(encoding='utf-8')
tree=ast.parse(source)
definitions={n.name:ast.get_source_segment(source,n) for n in tree.body
             if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
cells=[]
def md(s): cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
def functions(*names): code('\n\n\n'.join(definitions[n] for n in names))

md('''
# What comes after five songs?
### Spotify playlist continuation with track history, artist identity, and audio similarity

Given five consecutive tracks in a curated playlist, rank the catalogue for the track
in the sixth position. The central experiment measures the change in retrieval quality
when a model that already knows playlist history and artist identity gains access to
audio descriptors.

**Reading the notebook.** Sections 1-5 follow the assignment rubric: task, data,
modeling, evaluation, and related work with findings. Every reported result is produced
by the displayed code. The experiment is executed in Section 4; the result and its
interpretation appear there and in Section 5. This version does not export HTML.

To rerun, put `songs620.csv` beside the notebook, keep it in `data/`, or set
`SONGS620_PATH` to its location. A Desktop copy is also recognized. Run from the
folder containing the notebook. Install `requirements.txt` if needed.

[1 · Task](#part-1) · [2 · Data](#part-2) · [3 · Models](#part-3) ·
[4 · Evaluation](#part-4) · [5 · Findings and related work](#part-5)
''')
md(r'''
<a id="part-1"></a>
## 1 · Predictive task and validity criteria

**Observation and target.** Each test playlist supplies one opening query. Positions
0-4 are visible; position 5 supplies the single observed target. A playlist curator's
ordering supplies the label. Playback, skips, user satisfaction, and other plausible
continuations are unobserved. This experiment predicts the **immediate next track**,
rather than recovering arbitrary missing playlist members.

**Prediction setting.** A recommender can look up track IDs, artist IDs, and static
audio features for the entire snapshot catalogue, including newly encountered songs.
Playlist membership, track popularity, and transitions come only from training
playlists. This is a **known-catalogue, new-playlist** experiment. Treating every song
in the file as available at once is an explicit assumption because release and
playlist creation times are unavailable. Previously played tracks remain eligible
because curators can repeat them. A playlist is the context; no listener identities
are supplied, so this is not individual-user preference prediction.

**Primary comparison.** Compare the same logistic ranking pipeline with and without
audio, with artist features in both. The primary endpoint is **Hit Rate@10**, the
fraction of opening queries with the exact target in the top ten. **MRR@10** averages
reciprocal target rank for top-ten hits and zero otherwise. The target's median rank
shows movement below the cutoff. The full catalogue competes on every query.

**Baselines.** Uniform random ranking in expectation, training popularity, a last-artist
rule, item-item cosine similarity, a first-order Markov model, and audio similarity.
Three learned feature configurations separately measure history, artist, and audio.

**Validation plan.** Split playlist IDs 80/10/10 **before** constructing windows.
Choose inverse regularization strength C on validation openings. Freeze the fitted
models and evaluate test openings once for the primary comparison. No validation
playlists are folded into the final training statistics. Report results by zero versus
positive training support, a paired playlist-bootstrap interval for the primary
difference, and sensitivity checks for prefix order and position. Seed 42 is fixed;
its catalogue coverage was discussed during planning, so the split was not selected
blind to coverage. Model performance is not used to select or change it.

**Validity criteria.** Better than popularity is useful, but the central claim requires
an incremental gain from audio over history plus artist. A positive point estimate
with an interval spanning zero is insufficient evidence of a reliable gain. Better
median rank alone does not establish a benefit when only ten songs are shown.
''')
prefix=source[source.index('from __future__'):source.index('def find_dataset')]
prefix=prefix.replace('Path(__file__).resolve().parent','Path.cwd()')
code(prefix+"\nfrom IPython.display import display, Image, Markdown\n_thread_limit = threadpool_limits(limits=4)\n")
md('''
<a id="part-2"></a>
## 2 · Data, collection context, exploration, and preprocessing

### 2.1 What is known about this file

The supplied CSV links Spotify track URIs to playlist IDs (`pid`) and track positions
(`pos`), with artist/album metadata and audio descriptors. Its schema and playlist IDs
0-999 are consistent with an audio-enriched subset of the Million Playlist Dataset
(MPD), but **the exact derivation is unverified**: no collection script, source manifest,
sampling history, or enrichment date is supplied. The filename does not imply that
the file contains 620 tracks. No URL columns are fetched in this experiment.

Spotify's original MPD was released for playlist-continuation research and contains
one million playlists and over two million tracks [R1]. That describes the parent
resource, not the size or authenticated provenance of this local file. Curator order
is a useful label but is not evidence of actual listening behavior. No dates permit
chronological splitting or a check that all candidates existed at query time.

### 2.2 Preserve the sequence, separate the playlists

The loader checks nonmissing identifiers, integer positions, no duplicate `(pid,pos)`
pairs, contiguous positions from zero, and consistent per-track modeled metadata.
It sorts by playlist and position. Repeated `(pid,track)` memberships **remain in
the sequence**; binary co-occurrence statistics separately count each track once per
training playlist. This separates real repeat events from double-counting popularity.

The seed-42 permutation of sorted playlist IDs assigns the first 800 to training,
the next 100 to validation, and the last 100 to test. Every length-six training window
produces five inputs and one immediate-successor label. Validation and primary test
use only the opening window per eligible playlist. A playlist with only five tracks
has no sixth label and is reported as ineligible, rather than moved to another split.
''')
functions('find_dataset','queries_from_sequences','prepare_data')
code('''
data = prepare_data()
s = data['summary']
overview = pd.DataFrame({
    'quantity':['Sequence rows','Playlists','Catalogue tracks','Artists','Repeated memberships retained',
                'Unique memberships','Singleton tracks'],
    'value':[s['rows'],s['playlists'],s['tracks'],s['artists'],s['repeat_memberships'],
             s['unique_memberships'],s['singleton_tracks']]})
display(overview)
display(pd.Series(s['playlist_size'],name='Ordered playlist length').round(2).to_frame())
display(pd.DataFrame({'split':list(data['split']),
                      'playlists':[len(p) for p in data['split'].values()],
                      'learning_or_evaluation_queries':list(s['query_counts'].values()),
                      'ineligible_openings':list(s['excluded_short_openings'].values())}))
display(Image(filename=str(OUT/'data_overview.png')))
''')
code('''
display(Markdown(f"""
### 2.3 Observed structure and implications

There are **{s['rows']:,} ordered rows**, **{s['playlists']:,} playlists**,
**{s['tracks']:,} distinct tracks**, and **{s['artists']:,} artists.
The **{s['repeat_memberships']:,} repeated playlist-track memberships are retained**.
Only **{s['density']:.3%}** of possible playlist-track pairs are observed.
**{s['singleton_tracks']:,} tracks ({s['singleton_tracks']/s['tracks']:.1%})** appear
in just one playlist. A held-out playlist can therefore contain targets with no
training interactions even though their catalogue metadata are known.

Playlist lengths range from {int(s['playlist_size']['min'])} to
{int(s['playlist_size']['max'])} tracks, with median {s['playlist_size']['50%']:g}.
Training yields **{s['query_counts']['train']:,} overlapping windows**; validation
has **{s['query_counts']['validation']}** eligible opening queries and test has
**{s['query_counts']['test']}**. The validation exclusion of two short playlists
is a consequence of the fixed split, not a resampling decision.

Among adjacent **training** pairs, **{s['train_same_artist_transition_fraction']:.1%}**
share an artist. This motivates both a last-artist baseline and artist compatibility
features. Mean adjacent standardized-audio cosine is
**{s['train_adjacent_audio_cosine']:.3f}**, versus **{s['random_audio_cosine']:.3f}**
for random catalogue pairs. This motivates testing audio, but does not show that it
can identify the exact next track among thousands of plausible candidates.
"""))
display(pd.read_csv(OUT/'audio_summary.csv',index_col=0).round(3))
print('Missing values by column:', s['missing_by_column'])
print('Nonfinite raw audio values:', s['nonfinite_audio_values'])
''')
md('''
### 2.4 Audio features and the information boundary

Use ten continuous descriptors: danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, and duration. Duration is
log-transformed before standardization because its range is wide. A defensive median
imputer and standard scaler are fitted on **unique tracks appearing in training
playlists**, then applied to the entire catalogue. The supplied numerical inputs are
finite and complete. Zero-valued descriptors are retained as supplied; their upstream
meaning cannot be checked from this file. Key, mode, and time signature are omitted
from this predefined audio representation rather than imposing arbitrary ordinal
distances on those categories.

Track and artist URIs define stable categorical identities. No ordinal numeric ID
enters a linear predictor. IDs are used to look up matching identities, co-occurrences,
and transitions. Track/album names, album IDs, URL columns, and the CSV index column
are excluded from features; names appear only in readable examples. Position defines
the window but is not a predictor, so the model cannot memorize a position-5 label.

**Cross-fitting the relational features matters.** A training window's own playlist
contains its target, so computing its co-occurrence and transition features from that
playlist would expose part of its label. Four playlist folds avoid this: for windows
in a held fold, interaction statistics come only from the other 600 training
playlists. The audio scaler uses training-only static metadata. Validation and test
use statistics from all 800 training playlists, with no held-out playlist interactions.
This creates some feature-distribution shift from 600 to 800 support playlists,
which is a limitation of this economical cross-fitting design.
''')

md(r'''
<a id="part-3"></a>
## 3 · Models, optimization, and implementation

### 3.1 Task formulation and model families

For a history $h=(i_0,i_1,i_2,i_3,i_4)$ and candidate track $j$, construct features
$\phi(h,j)$ and score $s(h,j)$. Sort all catalogue tracks by decreasing score.
Ties are resolved by **training playlist support**, then lexicographically sorted
track URI. Previously played tracks are not masked. This common deterministic
tie rule is especially consequential for cold tracks and sparse Markov transitions.

| Method | Score and information used | Purpose |
|---|---|---|
| Uniform random | Expected HR@K = K / catalogue size | Chance reference, reported analytically |
| Popularity | Number of distinct training playlists containing candidate | Stronger than guessing; ignores history |
| Last artist | Candidate shares the fifth track's artist, followed by popularity tie-break | Tests how much simple artist persistence explains |
| Item cosine | Mean candidate cosine similarity to five history tracks, using binary training-playlist columns | Course-style neighborhood collaborative filtering |
| Markov | $P(j\mid i_4)$ from adjacent training transitions, smoothed toward training popularity | First-order sequential model |
| Audio similarity | Cosine of candidate standardized audio to mean history audio | Content-based comparison, including zero-support tracks |
| Logistic: history | Popularity, item cosines, Markov evidence, repeat fraction | Learned history-only control |
| Logistic: history + artist | Same features plus artist identity compatibility and transitions | Main control before audio |
| Logistic: history + artist + audio | Same history/artist features plus audio compatibility | Primary treatment |

For track transitions, use add-prior smoothing with total mass one:
$P(j\mid i)=(n_{ij}+p_0(j))/(n_i+1)$, where
$p_0(j)=(\mathrm{support}(j)+1)/(\sum_k\mathrm{support}(k)+N)$.
Unseen predecessor rows back off to that smoothed popularity prior. The same
construction uses artist transitions for artist features. These are statistics of
curator ordering, not observed playback transitions.

The course recommendation slides [L1] motivate cosine co-occurrence and ranking
evaluation; classification slides [L2] motivate logistic likelihood, L2 regularization,
and a distinct validation set. Temporal slides [L3] motivate sliding windows and
sequential recommendation. The random-playlist split does not turn curator position
into a genuine timestamp or establish chronological generalization.
''')
functions('TrainingStatistics','sparse_lookup','candidate_features')
code('''
display(pd.DataFrame({
    'group':['History','Artist','Audio'],
    'number_of_features':[len(HISTORY_NAMES),len(ARTIST_NAMES),len(AUDIO_NAMES)],
    'features':[', '.join(HISTORY_NAMES),', '.join(ARTIST_NAMES),', '.join(AUDIO_NAMES)]}))
''')
md(r'''
### 3.2 The controlled feature comparison

The history group has **7 features**: log candidate playlist support; mean, maximum,
and last-track item cosine; log smoothed next-track probability from the last track;
log mean transition probability from the five history tracks; and the fraction of
history positions containing the candidate itself.

The artist group adds **5 features**: same artist as the last track; fraction of
history positions matching the candidate artist; log artist playlist support; and
last/mean smoothed artist transition log probabilities. The model uses compatibility
with artist IDs, not a coefficient for each artist. This can generalize to a catalogue
artist that did not occur in a training window.

The audio group adds **14 features**: last-track and centroid cosine, last-track and
centroid Euclidean distance in standardized descriptor space, and ten absolute
descriptor differences from the last track. A linear coefficient can learn whether
closeness in a particular descriptor helps. Low-dimensional descriptors do not encode
lyrics, language, cultural theme, genre labels, or a learned audio embedding.

The comparison is nested: history uses 7, history+artist 12, and history+artist+audio
26 candidate features. All configurations share windows, sampled candidates, weights,
optimizer, and full-catalogue evaluation. Each feature configuration independently
selects C on validation HR@10; this compares equally tuned pipelines rather than
holding the fitted coefficients or regularization fixed.

### 3.3 Learning objective and negative sampling

For every training window, the observed successor is one positive example. Ten
distinct tracks are drawn uniformly without replacement from the rest of the
catalogue as pseudo-negatives. Only the exact target is excluded: a prefix track may
be a candidate again. Other unobserved tracks may also be plausible successors, so
pseudo-negative does not mean disliked. No validation or test labels are used to
sample or reject training candidates.

The sampled binary logistic objective is

$$-\sum_e w_e[y_e\log\sigma(\beta^T\tilde\phi_e+b)
+(1-y_e)\log(1-\sigma(\beta^T\tilde\phi_e+b))]
+\lambda\|\beta\|_2^2,$$

where $\tilde\phi$ is standardized using training examples and $w_e$ gives every
training playlist equal total loss weight despite different numbers of windows.
Weights are normalized to mean one. Scikit-learn uses inverse regularization C;
the exact scaling convention follows that implementation. This is a **pointwise
logistic candidate scorer used for ranking**, not a listwise softmax or a pairwise
BPR loss. Scores from this 1:10 sampled problem are not calibrated probabilities
of being the next song in the full catalogue.

The fixed C grid is {0.01, 0.1, 1, 10}. L-BFGS optimizes each model with up to
500 iterations. A convergence warning is treated as a failure rather than silently
accepted. Validation HR@10 chooses C; an exact tie chooses the smaller C in the
declared grid. MRR and median rank do not break the primary-metric tie. The chosen
models remain fitted on the original training examples; no test-guided refit occurs.
''')
functions('negative_candidates','training_examples','query_features','choose_models')
md('''
### 3.4 Tradeoffs, efficiency, and limitations

Artist and Markov rules are interpretable and cheap. Sparse transitions struggle with
unseen tracks; the popularity backoff often controls their rankings. Item cosine
aggregates five tracks and can exploit broader context, but it is also unsupported
for cold candidates. Audio can distinguish cold candidates yet may rank many
acoustically similar alternatives above the curator's exact choice.

The learned models combine these imperfect signals. L2 regularization, a small
predeclared feature set, and playlist-disjoint validation constrain overfitting.
Overlapping windows within training playlists are dependent; equal playlist weighting
reduces domination by long playlists, and evaluation uses one opening per playlist.
Uniform negatives are efficient but can make training easier than ranking among many
tracks by the same artist. Hard-negative experiments would be a separate, validation-
controlled study, not a change made after inspecting these test results.

Interactions, co-occurrences, and transitions use sparse matrices. Training candidates
require approximately 57 MB for the 26-column float32 feature array. Four-query
evaluation batches require about 15 MB for the final feature tensor, with additional
temporary arrays for audio and sparse lookups. The dense 100-query feature cube is
never retained. At full MPD scale even sparse co-occurrence construction can become
large; approximate candidate retrieval and a second-stage ranker would be appropriate.
This study avoids candidate pruning so every model competes over the same catalogue.

The models distinguish the last track from earlier history, but mean/max history
features are invariant to the order of the first four tracks. They are therefore
only partly order-sensitive. A higher-order Markov model or recurrent/attention model
could learn richer order, at greater data and optimization cost. Rating-MSE regression
is not the main comparison because no ratings are present; ordinary binary accuracy
would be dominated by the many unobserved pairs.
''')

md(r'''
<a id="part-4"></a>
## 4 · Evaluation and controlled comparisons

### 4.1 Exact ranks, metrics, and uncertainty

If the unique observed next track has one-based rank $r_p$ among all $N$ candidates:

$$HR@K=\frac1P\sum_p\mathbf1[r_p\le K],\qquad
MRR@K=\frac1P\sum_p\frac{\mathbf1[r_p\le K]}{r_p}.$$

The main display reports **HR@10**, **MRR@10**, and **median rank**. HR answers
whether a listener sees the curator's exact next song in a ten-song list; MRR rewards
putting a hit earlier in that list. Median rank captures typical retrieval position
below the cutoff, but may improve while the visible list does not. With one target,
recall@K equals HR@K and precision@K equals HR@K divided by K, so those redundant
metrics are omitted. Cutoffs 1, 5, 20, 50, and 100 show descriptive sensitivity to
list length; they do not replace the preselected primary endpoint.

Every query ranks **all catalogue tracks, including repeated prefix tracks**. There
are no sampled evaluation negatives or support filters. Training-time negative
sampling reduces optimization cost; it does not reduce the evaluation catalogue.
This distinction matters because sampled ranking metrics can change model comparisons
[R4]. Uniform random is reported analytically: HR@10 = 10/N, expected MRR@10 is
$\sum_{r=1}^{10}1/rN$, and the median rank is (N+1)/2.

The primary difference is HR@10(with audio) minus HR@10(without audio), with artist
features in both. Resample the 100 paired test playlists **10,000 times** to obtain
a 95% percentile bootstrap interval. Also report gained and lost hits and the exact
two-sided McNemar/binomial test on discordant outcomes. Equal total hits need not
mean the same playlists succeed. Only this audio comparison is the confirmatory
comparison; other model and subgroup results are descriptive.

The interval is conditional on these fitted models and this single split. It does
not include uncertainty from retraining, hyperparameter selection, a different
playlist sample, or a changed catalogue. Shared songs and artists may correlate
playlist outcomes. One hundred test playlists give limited power to detect small
incremental effects, so a nonsignificant result does not establish equivalence.
''')
functions('rank_target','ranking_metrics','evaluate_queries','paired_bootstrap',
          'summarize_and_plot','sensitivity_checks','run_experiment')
code('''
results = run_experiment(data)
table = results['table']
summary = results['summary']
primary = summary['primary']
display(Markdown('### 4.2 Validation grid and frozen configurations'))
display(results['validation'][['model','C','HR@10','MRR@10','Median rank','iterations','fit_seconds']].round(5))
display(pd.DataFrame(results['selection']).T[['C','HR@10','MRR@10']])
''')
code('''
display(Markdown('### 4.3 Test openings: full-catalogue ranking'))
random_row = summary['random_baseline']
view = table[['model','queries','Hits@10','HR@10','MRR@10','Median rank']].copy()
display(pd.concat([pd.DataFrame([random_row]).reindex(columns=view.columns),view],ignore_index=True).round(6))
display(Image(filename=str(OUT/'retrieval_comparison.png')))
display(pd.Series(primary,name='Audio minus no-audio, with artist in both').to_frame())
display(table[['model','HR@1','HR@5','HR@10','HR@20','HR@50','HR@100','MRR@10','MRR']].round(5))
display(Image(filename=str(OUT/'audio_ablation.png')))
''')
code('''
scores = table.set_index('model')
noaudio = scores.loc['History + artist']
audio = scores.loc['History + artist + audio']
lastartist = scores.loc['Last artist']
history = scores.loc['History']
direction = 'increases' if primary['delta_hr10'] > 0 else ('decreases' if primary['delta_hr10'] < 0 else 'does not change')
evidence = ('The exact paired test supports an audio gain at the 5% level.'
            if primary['exact_mcnemar_p'] < .05 and primary['delta_hr10'] > 0 else
            ('The exact paired test supports omitting audio at the 5% level.'
             if primary['exact_mcnemar_p'] < .05 and primary['delta_hr10'] < 0 else
             'The exact paired test does not establish a reliable difference at the 5% level; this does not establish equivalence.'))
display(Markdown(f"""
**Result in this sample.** The last-artist rule recovers **{int(lastartist['Hits@10'])} of
{int(lastartist['queries'])}** sixth songs. The learned history-only model recovers
**{int(history['Hits@10'])}**; history plus artist recovers **{int(noaudio['Hits@10'])}**;
and adding audio recovers **{int(audio['Hits@10'])}**. The supplied planning text's
23/24/24 figures are not hardcoded or treated as results of this implementation.

Adding audio **{direction}** observed HR@10 by **{primary['delta_hr10']:+.1%}**
(95% paired interval **[{primary['ci_low']:+.1%}, {primary['ci_high']:+.1%}]**).
There are **{primary['gained_hits']} gained** and **{primary['lost_hits']} lost** hits;
the exact paired test gives **p = {primary['exact_mcnemar_p']:.4f}**.
{evidence} MRR@10 changes from **{noaudio['MRR@10']:.4f}** to
**{audio['MRR@10']:.4f}**, so even equal HR would not imply equal ordering of hits.

Median target rank changes from **{noaudio['Median rank']:,.1f}** to
**{audio['Median rank']:,.1f}**. The median paired rank change is separately
**{primary['median_paired_rank_change']:,.1f}** (negative favors audio); a change in
the difference of medians is not the median of paired differences. These rank
summaries and the cutoff curve address a different question from top-ten success.
"""))
if primary['ci_low'] > 0 and primary['exact_mcnemar_p'] >= .05:
    display(Markdown("""
**Why the interval and test disagree here.** The plug-in percentile bootstrap
resamples only the observed query outcomes. With four gains and no losses it cannot
generate evidence of a loss, and its lower percentile can be positive even with very
few discordant queries. It can therefore be overconfident near this sparse boundary.
The exact paired test asks how surprising four wins out of four discordant outcomes
would be under equal win/loss chances; its two-sided p-value is 0.125. We retain both
requested diagnostics and do **not** use the positive bootstrap interval alone to
claim an established audio gain.
"""))
''')
code('''
display(Markdown('### 4.4 Zero versus positive training support'))
display(results['subgroup'][['model','support','queries','Hits@10','HR@10','MRR@10','Median rank']].round(5))
display(Markdown(f"""
**{summary['zero_support_targets']} of {summary['test_queries']}** test targets have
zero training playlist support. Their static artist and audio metadata are still
available. Track co-occurrence and transitions cannot provide positive item evidence
for them, although artist matching and content similarity may help. 'Unseen' below
means zero training interactions, not an unavailable or newly released track.

In these openings, the target shares the last artist in
**{summary['test_same_last_artist_fraction']:.1%}** of queries and repeats a prefix
track in **{summary['test_repeat_target_fraction']:.1%}**. These are test diagnostics,
not feature-selection criteria.
"""))
paired = pd.read_csv(OUT/'audio_paired_ranks.csv')
display(paired.assign(support=np.where(paired.track_support>0,'positive','zero')).groupby('support').agg(
    queries=('pid','size'),median_rank_without_audio=('rank_without_audio','median'),
    median_rank_with_audio=('rank_with_audio','median')))
''')
md('''
### 4.5 Sensitivity to order and position

These are predefined secondary stress checks with **frozen** models and the same
training statistics. Their results do not trigger tuning.

1. **Order:** reverse all five visible tracks, keeping the original sixth-track
   label. This changes which track is last as well as order; any difference measures
   sensitivity to that combined perturbation, not pure internal-order understanding.
   Also permute the first four while keeping the fifth fixed. The current models
   are invariant to that permutation by construction; verify it numerically.
2. **Position:** use positions 5-9 to predict position 10 in playlists long enough
   to supply it. Compare against openings from those **same eligible playlists**.
   Differences reflect both query position and target composition, and cannot be
   attributed causally to position alone. Each condition uses one query per playlist.
''')
code('''
sensitivity = results['sensitivity']
display(sensitivity[['setting','model','queries','HR@10','MRR@10','Median rank']].round(5))
original = results['details'].sort_values(['model','pid'])['rank'].to_numpy()
permuted = pd.read_csv(OUT/'sensitivity_first4_per_query.csv').sort_values(['model','pid'])['rank'].to_numpy()
assert np.array_equal(original,permuted)
print('Verified exact rank invariance when only the first four tracks are permuted.')
sens = sensitivity.set_index(['setting','model'])
display(Markdown(f"""
**Observed sensitivity.** Reversing all five tracks changes last-artist HR@10 from
**{lastartist['HR@10']:.1%}** to **{sens.loc[('Reversed five','Last artist'),'HR@10']:.1%}**;
the audio-augmented ranker changes from **{audio['HR@10']:.1%}** to
**{sens.loc[('Reversed five','History + artist + audio'),'HR@10']:.1%}**.
In contrast, permuting the first four while fixing the last changes no target ranks.
This supports sensitivity to the **last-track anchor**, not the ability to model
the internal order of the whole prefix.

There are **{int(sens.loc[('Positions 5-9 -> 10','History + artist'),'queries'])}**
eligible later-window playlists. On that matched subset, history-plus-artist HR@10
changes from **{sens.loc[('Opening, matched later playlists','History + artist'),'HR@10']:.1%}**
at the opening to **{sens.loc[('Positions 5-9 -> 10','History + artist'),'HR@10']:.1%}**
at the later position; the audio model changes from
**{sens.loc[('Opening, matched later playlists','History + artist + audio'),'HR@10']:.1%}**
to **{sens.loc[('Positions 5-9 -> 10','History + artist + audio'),'HR@10']:.1%}**.
Opening performance therefore does not transfer unchanged to later contexts.
"""))
''')
code('''
display(Markdown('### 4.6 Interpreting the fitted feature weights'))
coefficients = pd.read_csv(OUT/'coefficients.csv')
for name in MODEL_FEATURES:
    c = coefficients[coefficients.model==name].copy()
    c['absolute_weight'] = c.standardized_coefficient.abs()
    display(c.sort_values('absolute_weight',ascending=False).head(8).drop(columns='absolute_weight'))
''')
md('''
A standardized coefficient describes the conditional direction of the sampled
log-odds as that feature increases, holding the others fixed. Positive same-artist
weight is consistent with artist persistence; a negative distance coefficient is
consistent with preferring acoustic proximity. Correlated history, transition,
artist, and distance features can divide or reverse apparent importance, so weights
are not causal effects and are not directly comparable across differently tuned
models. The nested held-out comparison is stronger evidence of incremental value.
''')
code('''
display(Markdown('### 4.7 A fixed example, with readable tracks'))
example_pid = summary['example_pid']
opening = data['queries']['test'].set_index('pid').loc[example_pid]
track_ids = opening[[f'h{k}' for k in range(5)]].to_numpy(dtype=int)
display(data['tracks'].iloc[track_ids][['track_name','artist_name']].assign(position=np.arange(5)))
display(data['tracks'].iloc[[int(opening.target)]][['track_name','artist_name']].assign(position=5))
example = pd.read_csv(OUT/'test_top10.csv')
display(example[(example.pid==example_pid)&example.model.isin(
    ['Last artist','History + artist','History + artist + audio'])])
print('Example playlist chosen by smallest test pid:', example_pid)
''')

md('''
<a id="part-5"></a>
## 5 · Findings, related work, and conclusions

### 5.1 What prior work contributes

**Dataset and continuation context.** Spotify's MPD announcement [R1] establishes
the original dataset's continuation purpose. Zamani et al.'s challenge analysis [R2]
studies recommendation of fitting additions from seed playlists and metadata. The
present task is narrower: one exact immediate successor after five ordered tracks,
100 test openings, and a much smaller catalogue. Official MPD challenge numbers
must not be used as a directly comparable benchmark.

**Context versus order.** Vall et al. [R3] compare popularity, collaborative filtering,
and recurrent models on curated playlists. They find useful song-context signal,
while order is not crucial in their reported experiments. This motivates the
history comparisons and order stress checks here. Our rankers encode the last
track separately but do not learn the order of the earlier four tracks, so they
cannot establish that richer sequential modeling is unnecessary.

**Audio and hybrid recommendation.** Ferraro et al. [R5] combine audio/title-aware
factorization with co-occurrence evidence through ranking fusion. Their hybrid
motivates testing content alongside interaction features. The current features
are low-dimensional Spotify descriptors; they omit playlist titles and learned
audio representations. Weak incremental audio value here would not refute a richer
hybrid, especially with a different catalogue and relevance objective.

**Evaluation choice.** Krichene and Rendle [R4] show that sampled recommendation
metrics can change the relative comparison between algorithms. That supports full-
catalogue evaluation here. Sampled pseudo-negatives remain an optimization choice,
explicitly separate from evaluation candidates.

These are relevant established research comparisons, not a claim to reproduce a
current state-of-the-art system. A stronger next experiment could compare a sequential
attention/recurrent model and an audio-embedding hybrid under this same fixed protocol,
with enough playlist data and validation capacity to tune them fairly.
''')
code('''
display(Markdown('### 5.2 What the current experiment supports'))
best_model = scores['HR@10'].idxmax()
best_score = scores.loc[best_model,'HR@10']
zero = results['subgroup'].query("support == 'zero'").set_index('model')
display(Markdown(f"""
The strongest observed top-ten result is **{best_model}**, with
**HR@10 = {best_score:.3f}**. This is a descriptive comparison across the reported
methods, not a new test-driven model-selection step. The main audio comparison
is between the two predeclared artist-aware pipelines, not between whichever two
models happen to rank highest on test.

Adding artist features changes the learned history-only model from
**{int(history['Hits@10'])}** to **{int(noaudio['Hits@10'])}** hits. The last-artist rule
alone obtains **{int(lastartist['Hits@10'])}** hits. Their proximity or difference
helps assess how much complexity buys beyond artist persistence. This is an
ablation association on held-out playlists, not a causal claim about artist identity.

Audio changes HR@10 by **{primary['delta_hr10']:+.1%}** with paired interval
**[{primary['ci_low']:+.1%}, {primary['ci_high']:+.1%}]**. {evidence}
For zero-support targets, median rank is
**{zero.loc['History + artist','Median rank']:,.1f}** without audio and
**{zero.loc['History + artist + audio','Median rank']:,.1f}** with audio.
The sign and size of that shift should be read alongside top-ten subgroup hits:
retrieval below the cutoff may matter for a large candidate pool, but does not
automatically help a listener who sees only ten songs.
"""))
''')
md('''
### 5.3 Practical interpretation and limits

Choose the objective before choosing the method. A ten-song interface should be
judged mainly by HR@10 and ordering of those hits. A later-stage candidate generator
may value rank improvements deeper in the catalogue, especially for zero-support
targets. These roles can favor different feature sets. A bigger list or better
median rank should not be presented as a top-ten gain.

This study is limited by uncertain file provenance, one coverage-aware split,
98 validation and 100 test openings, an assumed simultaneously available catalogue,
incomplete relevance labels, and sampled training candidates. Opening queries may
behave differently from arbitrary positions. There are no users, playback logs, or
dates with which to establish listener satisfaction or chronological generalization.
The out-of-fold training feature support differs from final support, and short
same-artist runs can reward artist rules without measuring musical variety.

The next evidence to collect is a larger playlist-disjoint test sample, repeated
predeclared splits with paired uncertainty, more realistic hard negatives, richer
audio/text representations, and a genuinely order-sensitive model. A listening study
or behavior logs would be needed to judge pleasant alternatives rather than exact
agreement with the curator's sixth song. These follow-up experiments are proposals;
none is reported as an achieved result.

### 5.4 References

- **[L1] Julian McAuley, _Recommender Systems_, course slides.**
  [Supplied recommendation slides](https://drive.google.com/file/d/1VAt1C5Za5JCBXUkeRbqkH8jajgTdfmnl/view).
  Cosine collaborative filtering, cold start, and ranking evaluation.
- **[L2] Julian McAuley, _Classification_, course slides.**
  [Supplied classification slides](https://drive.google.com/file/d/1fUrZCpQ6Sx92blcn6NshOPEWxb27Jp6k/view).
  Logistic learning, regularization, and validation.
- **[L3] Julian McAuley, _Temporal data mining_, course slides.**
  [Supplied temporal slides](https://drive.google.com/file/d/1YQdxP_boLqM24MEiXy65ii3Utiqp55R3/view).
  Sliding windows and sequential recommendation.
- **[R1] Ching-Wei Chen (2020), _The Million Playlist Dataset... Remastered_.**
  [Spotify Research](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered).
- **[R2] Hamed Zamani, Markus Schedl, Paul Lamere, and Ching-Wei Chen (2019),
  _An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic
  Music Playlist Continuation_.** [Paper](https://arxiv.org/abs/1810.01520).
- **[R3] Andreu Vall, Massimo Quadrana, Markus Schedl, and Gerhard Widmer (2018),
  _The Importance of Song Context and Song Order in Automated Music Playlist
  Generation_.** [Paper](https://arxiv.org/abs/1807.04690).
- **[R4] Walid Krichene and Steffen Rendle (2021), _On Sampled Metrics for Item
  Recommendation (Extended Abstract)_.** [IJCAI proceedings](https://ijcai.org/proceedings/2021/651).
- **[R5] Andres Ferraro, Dmitry Bogdanov, Jisang Yoon, KwangSeob Kim, and Xavier Serra
  (2018; arXiv 2019), _Automatic playlist continuation using a hybrid recommender
  system combining features from text and audio_.** [Paper](https://arxiv.org/abs/1901.00450).

### Reproducibility and deliverable scope

`output/next_song/` contains the playlist split, query windows, catalogue vocabulary,
cross-fitting fold assignments, full validation grid, frozen C choices, coefficients,
per-query ranks, top-ten lists, paired audio ranks, sensitivity results, figures,
environment versions, and dataset checksum. `continuation_experiment.py` executes
the same experiment from the command line. This notebook contains its functions
and produces its outputs without importing that file.

`verify_continuation.py` independently checks immediate-successor windows with repeats,
exclusion of held playlists from support statistics, deterministic ties and repeat
eligibility, hand-calculated retrieval metrics, and paired hit accounting. Numerical
thread pools are limited to four. The preserved earlier missing-member experiment
and `workbook.html` concern the former task; they are not this notebook's results.
No HTML, video recording, upload, or peer-grading submission is produced in this step.
''')
code('''
display(pd.Series(summary['environment'],name='version').to_frame())
print('Dataset SHA-256:', summary['dataset_sha256'])
print('Full catalogue size:', len(data['tracks']))
print('Primary bootstrap draws:', summary['primary']['bootstrap_draws'])
assert set(data['split']['train']).isdisjoint(data['split']['validation'])
assert set(data['split']['train']).isdisjoint(data['split']['test'])
assert set(data['split']['validation']).isdisjoint(data['split']['test'])
print('Verified playlist-disjoint splits. All five sections completed; no HTML exported.')
''')

notebook=nbformat.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.14'},
    'title':'What comes after five songs?'})
if __name__=='__main__':
    path=ROOT/'assignment_2.ipynb'
    nbformat.write(notebook,path)
    print(f'Created {len(cells)} cells; executing all five sections...',flush=True)
    client=NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook,path)
    print('Executed assignment_2.ipynb. No HTML export performed.',flush=True)
