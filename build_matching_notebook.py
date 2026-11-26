"""Build and execute a self-contained five-section playlist-matching notebook."""
from pathlib import Path
import ast
import os
import textwrap
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
for name, folder in [('JUPYTER_RUNTIME_DIR', 'jupyter'), ('JUPYTER_CONFIG_DIR', 'jupyter_config'),
                     ('IPYTHONDIR', 'ipython'), ('MPLCONFIGDIR', 'matplotlib')]:
    os.environ[name] = str(ROOT / 'tmp' / folder)
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
source = (ROOT / 'matching_experiment.py').read_text(encoding='utf-8')
tree = ast.parse(source)
definitions = {n.name: ast.get_source_segment(source, n) for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))


def functions(*names):
    code('\n\n\n'.join(definitions[n] for n in names))


md('''
# Which song fits this playlist?
### Choosing between two candidates using five example songs

**Research question:** Given five songs from a playlist, can we identify which of two
candidate songs also belongs to that playlist?

We compare familiar rules—popularity, artist overlap, and audio similarity—with
logistic regression. One candidate is a hidden playlist member; the other is sampled
uniformly from songs outside the playlist. Random guessing scores **50%**.

Sections 1–5 cover the task, data, models, evaluation, and findings with related work.
The notebook contains the complete implementation and executed results. To rerun,
keep `data/songs620.csv` here, put the CSV beside this notebook, or set `SONGS620_PATH`.
Install `requirements.txt`, then run all cells from the notebook's folder.
''')
md('''
## 1 · Predictive task

For each eligible playlist, sample six **distinct** songs without replacement.
Five form the visible context, and the sixth is the hidden member. Sample a second
candidate uniformly from all distinct catalogue tracks absent from the **entire**
playlist, and randomly place the hidden member in candidate A or B. Predict which
candidate is the member. Order and original track positions play no role.

**Example:** five songs from a playlist → candidate A versus candidate B → choose A or B.
The label is observed playlist membership, not a judgment that the other song sounds bad.

**Evaluation plan:** fix seed 42 and split whole playlists 80%/10%/10% into training,
validation, and test before making examples. Draw ten examples per eligible playlist
in each split. Examples can reuse songs within a playlist, so uncertainty is measured
by resampling **playlists**, not by treating every query as independent.

Choose regularization and the primary learned model on validation accuracy, freeze
these choices, then evaluate test accuracy. Ties get half credit: the expected result
of a fair coin flip. All eligible playlists supply ten queries, giving them equal weight.

**What success means:** accuracy above the 50% random baseline, with a playlist-bootstrap
interval and paired comparisons against audio similarity. A higher score alone does
not establish that learning improves on a simple rule.

**Scope:** this is a known-catalogue, new-playlist task. Static metadata may be looked up
for every candidate. Playlist-derived popularity and fitted preprocessing use training
playlists only. Track overlap between splits is allowed; playlist overlap is not.
''')
prefix = source[source.index('from __future__'):source.index('def find_dataset')]
prefix = prefix.replace('Path(__file__).resolve().parent', 'Path.cwd()')
code(prefix + '\nfrom IPython.display import display, Image, Markdown\n_thread_limit = threadpool_limits(limits=4)')
md('''
## 2 · Data and preparation

The local `songs620.csv` contains playlist IDs, track and artist URIs, positions,
and numeric audio descriptors. Playlist membership supplies labels; danceability,
energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence,
tempo, and duration supply audio predictors. We do not fetch Spotify URLs.

Spotify's [Million Playlist Dataset introduction](https://engineering.atspotify.com/2018/05/introducing-the-million-playlist-dataset-and-recsys-challenge-2018)
describes a public-playlist research resource. The local CSV has a compatible schema,
but its exact sampling and audio-enrichment history are **unverified**. We therefore
describe the supplied file directly rather than assume its provenance. It has no
listener IDs, satisfaction labels, or timestamps for a chronological split.

Deduplicate `(playlist, track)` memberships so a repeated recording does not count as
two distinct songs. Identify recordings by URI, not name. Reject missing or conflicting
modeled metadata and nonfinite audio values. Playlists with fewer than six distinct
tracks stay in their original split but cannot supply examples.

Log-transform tempo and duration using `log1p`, then standardize the ten audio features
using **distinct tracks in training playlists only**. These transformations are fitted
once and applied unchanged to validation and test candidates. Popularity counts the
number of training playlists containing each track.
''')
functions('find_dataset', 'prepare_data', 'make_queries')
code('''
data = prepare_data()
display(pd.Series(data['summary']).drop(['eligible_playlists', 'split_playlists', 'dataset_sha256']).to_frame('value'))
display(pd.DataFrame({'all_playlists': data['summary']['split_playlists'],
                      'eligible_playlists': data['summary']['eligible_playlists']}))
display(data['tracks'][['track_name', 'artist_name'] + AUDIO].head())
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist([len(v) for v in data['members'].values()], bins=30, color='#4786b8')
axes[0].set(xlabel='Distinct songs per playlist', ylabel='Playlists', title='Playlist sizes')
counts = data['raw'].groupby('item').pid.nunique()
axes[1].hist(counts, bins=30, color='#238b45')
axes[1].set(xlabel='Playlists containing a track', ylabel='Distinct tracks', title='Track support')
axes[1].set_yscale('log')
fig.tight_layout()
display(fig)
plt.close(fig)
print(f"{data['summary']['singleton_tracks'] / len(data['tracks']):.1%} of tracks occur in just one playlist.")
''')
md('''
Many songs appear in only one playlist, limiting identity-based retrieval. Static audio
features still describe these songs. Playlist matching between two candidates reduces
the search burden while retaining an observed behavioral label.

The [earlier Spotify project presentation](https://docs.google.com/presentation/d/1er6tXVBatdB-UJQ5t9p4c7T3HA4U4P42Cuk8XslenCA/edit)
in Google Drive motivates grouping and recommending songs using their features. It is
background context, not an additional source of training labels or measured accuracy.
''')
md('''
## 3 · Models and fitting

Each model gives candidate A and B a compatibility score and chooses the higher one.

| Model | Compatibility rule |
|---|---|
| Random | Fair coin; expected accuracy 50% |
| Popularity | Number of training playlists containing the candidate |
| Artist overlap | Fraction of the five context songs sharing the candidate's artist |
| Audio distance | Negative sum of absolute standardized differences from the context's mean audio vector |
| Logistic: audio | Learn weights for ten per-feature closeness values and distance to the nearest context song |
| Logistic: audio + artist | Add the artist-overlap feature to the same learned model |

For each audio dimension, closeness is the negative absolute difference between the
candidate and the five-song mean. Nearest-song closeness is negative Euclidean distance
to the closest of the five songs. Higher values mean greater similarity.

**Simple explanation of logistic regression:** learn how much weight to give similarities
such as energy, tempo, and artist overlap when choosing between the two songs.

Subtract B's features from A's features, then fit logistic regression to predict whether
A is the member. Include each training pair and its swapped version with the opposite
label; omit the intercept. Reversing A and B therefore reverses the score. Scale feature
differences using training examples only. There are no track-ID, album-ID, playlist-ID,
or co-occurrence predictors, so a training playlist cannot leak its membership through
its own historical counts into a learned feature.

Test C ∈ {0.01, 0.1, 1, 10} for each learned model. Choose the highest validation
accuracy, breaking ties toward smaller C. Choose the primary learned model by validation
accuracy, breaking ties toward fewer features. Keep the original training fit after
selection. Coefficients describe associations under this sampling scheme, not causes.
''')
functions('candidate_features', 'query_features', 'baseline_scores', 'correct_credit', 'fit_models')
md('''
## 4 · Evaluation on held-out playlists

Run the fixed experiment below. Accuracy is the fraction of correctly chosen members,
with half credit for exact score ties. The uncertainty interval resamples the 100 held-out
playlists with replacement 10,000 times. Ten queries from the same playlist remain together.
Paired intervals compare models on exactly the same queries and playlists.

All candidate pairs, split assignments, validation scores, frozen model choices,
coefficients, test predictions, subgroup results, and dataset checksum are saved under
`output/playlist_matching/`. Validation chooses the primary model; test results do not.
''')
functions('bootstrap_interval', 'evaluate', 'paired_comparison', 'run_experiment')
code('''
results = run_experiment(data)
summary = results['summary']
display(pd.Series(summary['query_counts'], name='queries').to_frame())
display(results['grid'])
print('Validation-selected primary model:', summary['selection']['primary_model'])
display(results['metrics'].style.format({'accuracy': '{:.1%}', 'ci_low': '{:.1%}',
                                        'ci_high': '{:.1%}', 'tie_fraction': '{:.1%}'}))
display(Image(filename=str(OUT / 'accuracy.png')))
display(pd.DataFrame(summary['comparisons']))
''')
code('''
primary = summary['selection']['primary_model']
row = results['metrics'].set_index('model').loc[primary]
baseline = results['metrics'].set_index('model').loc['Audio distance']
comparison = summary['comparisons'][1]
gain = ('The interval supports an improvement over the audio rule.' if comparison['ci_low'] > 0
        else 'The interval does not establish an improvement over the audio rule.'
        if comparison['ci_high'] >= 0 else 'The audio rule performs better in this comparison.')
display(Markdown(f"""**Measured result.** The validation-selected **{primary}** scores
**{row['accuracy']:.1%}** on held-out playlists (95% playlist-bootstrap interval:
{row['ci_low']:.1%}–{row['ci_high']:.1%}). Random guessing scores 50% in expectation.
The simple audio-distance rule scores **{baseline['accuracy']:.1%}**.
The learned model's paired difference from that rule is **{comparison['difference']*100:.1f}
percentage points**, with interval [{comparison['ci_low']*100:.1f}, {comparison['ci_high']*100:.1f}].
{gain}"""))
display(results['subgroups'])
display(results['coefficients'].query('model == @primary').sort_values('coefficient', ascending=False))
''')
md('''
The subgroup table separates queries whose hidden member appeared in a training playlist
from queries whose member did not. Within each subgroup, accuracy first averages within
playlist and then across playlists that have such queries. These are descriptive checks;
they do not select another model. Coefficients use one standard deviation of training
pair differences; correlated features can share or reverse weights.
''')
code('''
# Show the first test example, selected by playlist/query ID rather than outcome.
q = results['queries']['test'].sort_values(['pid', 'query']).iloc[0]
seed_items = q[[f'h{i}' for i in range(5)]].to_numpy(dtype=int)
display(data['tracks'].iloc[seed_items][['track_name', 'artist_name']].assign(role='Context'))
example = data['tracks'].iloc[[int(q.candidate_a), int(q.candidate_b)]][['track_name', 'artist_name']].copy()
example.index = ['A', 'B']
example['actual_member'] = [bool(q.label), not bool(q.label)]
display(example)
details = results['details']
prediction = details[(details.model == primary) & (details.pid == q['pid']) &
                     (details['query'] == q['query'])]
score = float(prediction.score_a_minus_b.iloc[0])
print('Model chooses:', 'A' if score > 0 else 'B' if score < 0 else 'Tie: fair coin')
''')
md('''
## 5 · Findings, related work, and limits

The measured result and confidence intervals above answer the research question directly.
Compare the learned model to both random guessing and the audio rule before claiming
that a more complex model is useful. Artist overlap may help explain shared performers,
while audio similarity remains available for recordings absent from training playlists.
The paired audio-plus-artist comparison measures the observed incremental benefit of
artist overlap rather than assuming it helps.

**How this relates to existing work.** Spotify's
[2018 playlist-continuation challenge](https://research.atspotify.com/publications/recsys-challenge-2018-automatic-music-playlist-continuation)
studied recommending tracks for existing playlists. Our task shares that motivation but
asks a narrower question: distinguish one observed member from one sampled nonmember.
Its catalogue-retrieval metrics and this notebook's two-candidate accuracy cannot be
compared numerically. The Google Drive presentation linked in Section 2 provides the
project's earlier feature-based grouping and recommendation context; it does not
validate the results of this new experiment.

**Limits that matter when explaining accuracy:**

- A randomly sampled nonmember may be easy to distinguish. Accuracy applies to this
  exact candidate sampler; it does not establish performance against carefully selected
  similar alternatives or over the entire catalogue.
- Absence from a playlist does not imply dislike. The label measures recorded membership,
  not musical quality, listener satisfaction, or whether a new recommendation is welcome.
- Random five-song contexts test playlist compatibility, not the immediate next song.
  Deduplication and context sampling intentionally discard order.
- Playlists are held out, but tracks and artists can overlap across splits. Metadata for
  all candidates is assumed available. No dates support claims about future playlists.
- Different playlist IDs could still share very similar content. The playlist split does
  not guarantee independence of curators or eliminate copied playlists.
- Ten queries per playlist share context and labels. Playlist-bootstrap intervals account
  for this clustering, conditional on the fixed split, sampled queries, and fitted model;
  they do not capture all variation from retraining or another negative sampler.
- The test split has been explored in the earlier next-song task. This is a new task and
  a fixed protocol, not a claim that the underlying data has never been inspected.

**Presentation-ready explanation:** “We show five songs from a playlist and ask our model
to choose which of two other songs was also in it. It compares how similar the songs
sound and whether they share artists. We test it on playlists excluded from training.”

For a stronger follow-up, freeze this model and evaluate a separately defined set of
more similar nonmembers. Do not quietly replace the current test sampler to improve
the headline accuracy.
''')
code('''
split_sets = [set(ids) for ids in data['split'].values()]
assert all(split_sets[i].isdisjoint(split_sets[j]) for i in range(3) for j in range(i))
for split, queries in results['queries'].items():
    assert set(queries.pid) <= set(data['split'][split])
    for q in queries.itertuples(index=False):
        context = {getattr(q, f'h{i}') for i in range(5)}
        member = q.candidate_a if q.label else q.candidate_b
        outsider = q.candidate_b if q.label else q.candidate_a
        assert len(context) == 5 and member not in context
        assert context | {member} <= set(data['members'][q.pid])
        assert outsider not in data['members'][q.pid]
print('Verified: disjoint playlist splits, distinct contexts, hidden members, and true nonmembers.')
display(pd.Series(summary['environment'], name='version').to_frame())
print('Dataset SHA-256:', summary['data']['dataset_sha256'])
''')


if __name__ == '__main__':
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'title': 'Which song fits this playlist?'})
    path = ROOT / 'assignment_2.ipynb'
    nbformat.write(notebook, path)
    print(f'Created {len(cells)} cells; executing...', flush=True)
    try:
        NotebookClient(notebook, timeout=1200, kernel_name='python3',
                       resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook, path)
    nbformat.write(notebook, ROOT / 'cse158_assignment2.ipynb')
    print('Executed assignment_2.ipynb.', flush=True)
