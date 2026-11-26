"""Independent membership, symmetry, split, and metric checks."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from matching_experiment import (make_queries, candidate_features, correct_credit,
                                 fit_models, bootstrap_interval, MODEL_COLUMNS)


class MatchingChecks(unittest.TestCase):
    def test_sampler_excludes_whole_playlist_and_hides_distinct_member(self):
        members = {1: np.arange(8), 2: np.arange(5)}
        queries = make_queries(members, [1, 2], 12, seed=9, per_playlist=50)
        self.assertEqual(len(queries), 50)
        self.assertEqual(set(queries.pid), {1})
        for q in queries.itertuples():
            context = {getattr(q, f'h{i}') for i in range(5)}
            positive = q.candidate_a if q.label else q.candidate_b
            negative = q.candidate_b if q.label else q.candidate_a
            self.assertEqual(len(context | {positive}), 6)
            self.assertTrue(context | {positive} <= set(members[q.pid]))
            self.assertNotIn(negative, members[q.pid])
        pd.testing.assert_frame_equal(queries, make_queries(members, [1, 2], 12, 9, 50))

    def test_features_match_hand_calculation(self):
        data = {'z': np.array([[0., 0.], [2., 2.], [1., 1.]]),
                'artists': np.array(['a', 'b', 'a'])}
        result = candidate_features(data, np.array([[0, 1]]), np.array([2]))
        np.testing.assert_allclose(result, [[0, 0, -np.sqrt(2), .5]])

    def test_accuracy_counts_ties_fairly(self):
        np.testing.assert_array_equal(correct_credit([2, -2, 0, 1], [1, 0, 1, 0]), [1, 1, .5, 0])
        self.assertEqual(bootstrap_interval([.5] * 5), [.5, .5])

    def test_learned_scores_reverse_when_candidates_swap(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(80, 12))
        y = (x[:, 0] > 0).astype(int)
        models, _, _, _ = fit_models(x[:60], y[:60], x[60:], y[60:])
        for name, model in models.items():
            features = x[60:, MODEL_COLUMNS[name]]
            np.testing.assert_allclose(model.decision_function(features),
                                       -model.decision_function(-features), atol=1e-12)

    def test_saved_results_match_raw_membership_and_metrics(self):
        root = Path(__file__).resolve().parent
        out = root / 'output' / 'playlist_matching'
        raw = pd.read_csv(root / 'data' / 'songs620.csv')
        catalogue = pd.read_csv(out / 'catalogue.csv')
        lookup = dict(zip(catalogue.track_uri, catalogue.item))
        members = raw.assign(item=raw.track_uri.map(lookup)).groupby('pid').item.apply(set).to_dict()
        split = pd.read_csv(out / 'playlist_split.csv')
        self.assertEqual(split.pid.nunique(), len(split))
        self.assertEqual(split.groupby('split').size().to_dict(), {'train': 800, 'validation': 100, 'test': 100})
        for name in ['train', 'validation', 'test']:
            q = pd.read_csv(out / f'{name}_queries.csv')
            self.assertTrue(set(q.pid) <= set(split.loc[split.split == name, 'pid']))
            for row in q.itertuples():
                context = {getattr(row, f'h{i}') for i in range(5)}
                positive = row.candidate_a if row.label else row.candidate_b
                negative = row.candidate_b if row.label else row.candidate_a
                self.assertEqual(len(context | {positive}), 6)
                self.assertTrue(context | {positive} <= members[row.pid])
                self.assertNotIn(negative, members[row.pid])
        details = pd.read_csv(out / 'test_per_query.csv')
        metrics = pd.read_csv(out / 'test_metrics.csv').set_index('model')
        for name, rows in details.groupby('model'):
            np.testing.assert_allclose(rows.correct_credit, correct_credit(rows.score_a_minus_b, rows.label))
            self.assertAlmostEqual(rows.groupby('pid').correct_credit.mean().mean(), metrics.loc[name, 'accuracy'])
            self.assertEqual(len(rows), 1000)


if __name__ == '__main__':
    unittest.main()
