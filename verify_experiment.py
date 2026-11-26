"""Check ranking, leakage boundaries, and metric arithmetic on independent examples."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from experiment import Ranker, evaluate, top_items


class ExperimentChecks(unittest.TestCase):
    def test_seed_exclusion_and_score_ties(self):
        scores = np.array([10., 4., 4., 3., 4.])
        seen = np.array([True, False, False, False, False])
        pop = np.array([100., 1., 2., 0., 2.])
        np.testing.assert_array_equal(top_items(scores, seen, pop, 3), [2,4,1])

    def test_knn_self_neighbor_excluded(self):
        b = csr_matrix(np.array([[1,1,0],[1,0,1],[0,0,1]],dtype=np.float32))
        model = Ranker("Playlist-kNN", {"neighbors":1,"power":1}).fit(b, None)
        np.testing.assert_array_equal(model.w.diagonal(), 0)
        # Playlist 0's sole neighbor is playlist 1; its unseen track 2 must win.
        self.assertGreater(model.scores(0,1)[0,2], 0)

    def test_metrics_against_hand_calculation(self):
        ranked = np.arange(50)[None,:]
        truth = csr_matrix(([1.,1.],([0,0],[0,2])),shape=(1,55))
        fit = csr_matrix(([1.],([0],[54])),shape=(1,55))
        tracks = pd.DataFrame({"artist_uri":[str(i) for i in range(55)]})
        m = evaluate(ranked,truth,fit,tracks).iloc[0]
        self.assertAlmostEqual(m["Recall@20"], 1.)
        self.assertAlmostEqual(m["Precision@20"], .1)
        self.assertAlmostEqual(m["NDCG@20"], 1.5/(1+1/np.log2(3)))

    def test_saved_manifest_and_outputs(self):
        out = Path(__file__).resolve().parent / "output"
        split = pd.read_csv(out/"split_manifest.csv")
        self.assertFalse(split.duplicated(["u","i"]).any())
        summary = json.loads((out/"experiment_summary.json").read_text())
        self.assertEqual(len(split),sum(summary["split_counts"].values()))
        fit_pairs = split[split.split != "test"]
        for name in ["popularity","playlist-knn","puresvd","audio_centroid"]:
            ranks = np.load(out/(name+"_top50.npy"))
            for u,group in fit_pairs.groupby("u"):
                self.assertEqual(len(np.unique(ranks[u])),50)
                self.assertFalse(np.isin(ranks[u],group.i).any())


if __name__ == "__main__":
    unittest.main()
