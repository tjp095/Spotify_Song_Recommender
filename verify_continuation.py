"""Checks independent of model scores: chronology, support boundaries, ties, and metrics."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from continuation_experiment import (TrainingStatistics, candidate_features, queries_from_sequences,
    rank_target, ranking_metrics, paired_bootstrap, FEATURE_NAMES)


class ContinuationChecks(unittest.TestCase):
    def test_windows_keep_repeats_and_predict_immediate_successor(self):
        seq={7:np.array([0,1,0,2,3,0,4])}
        q=queries_from_sequences(seq,[7],all_windows=True)
        self.assertEqual(q.target.to_list(),[0,4])
        self.assertEqual(q.iloc[0][['h0','h1','h2','h3','h4']].to_list(),[0,1,0,2,3])

    def test_statistical_support_excludes_held_playlist(self):
        d={'tracks':list(range(7)),'n_artists':3,'artist_ids':np.array([0,0,1,1,2,2,2]),
           'sequences':{1:np.array([0,1,2,3,4,5]),2:np.array([6,6,6,6,6,6])}}
        s=TrainingStatistics(d,[1])
        self.assertEqual(s.pop[6],0)
        self.assertEqual(s.transition[6,6],0)
        self.assertEqual(s.transition[0,1],1)
        d.update(z=np.ones((7,10),dtype=np.float32),zn=np.ones((7,10),dtype=np.float32)/np.sqrt(10))
        f=candidate_features(d,s,np.array([[0,1,2,3,4]]),np.arange(7))
        self.assertEqual(f.shape,(1,7,len(FEATURE_NAMES)))
        self.assertEqual(f[0,6,0],0)

    def test_repeat_target_is_eligible_and_ties_are_deterministic(self):
        # No observed-history masking: repeated track zero may win again.
        self.assertEqual(rank_target(np.array([4.,1.,0.]),0,np.array([0,0,0])),1)
        score=np.array([2.,2.,2.,1.]); pop=np.array([1,2,2,9])
        self.assertEqual(rank_target(score,2,pop),2)
        self.assertEqual(rank_target(score,0,pop),3)

    def test_metrics_use_single_next_target(self):
        m=ranking_metrics([1,2,11,100])
        self.assertEqual(m['HR@10'],.5)
        self.assertEqual(m['MRR@10'],.375)
        self.assertEqual(m['Median rank'],6.5)

    def test_equal_hit_counts_can_have_discordant_queries(self):
        p=paired_bootstrap([1,20,30],[20,1,40])
        self.assertEqual(p['delta_hr10'],0)
        self.assertEqual(p['gained_hits'],1)
        self.assertEqual(p['lost_hits'],1)
        self.assertEqual(p['exact_mcnemar_p'],1)

    def test_saved_protocol_and_ranking_accounting(self):
        out=Path(__file__).resolve().parent/'output'/'next_song'
        if not (out/'test_per_query.csv').exists():
            self.skipTest('Run the experiment before checking saved results.')
        split=pd.read_csv(out/'playlist_split.csv')
        self.assertEqual(split.pid.nunique(),1000)
        self.assertEqual(split.groupby('split').size().to_dict(),{'test':100,'train':800,'validation':100})
        test_pids=set(split.loc[split.split=='test','pid'])
        details=pd.read_csv(out/'test_per_query.csv')
        self.assertEqual(set(details.pid),test_pids)
        self.assertTrue(details.groupby('model').size().eq(100).all())
        self.assertTrue(details.groupby('pid').target.nunique().eq(1).all())
        self.assertTrue(details['rank'].between(1,36203).all())
        top=pd.read_csv(out/'test_top10.csv')
        self.assertTrue(top.groupby(['pid','model']).item.nunique().eq(10).all())
        hit_from_list=top.groupby(['pid','model']).is_target.sum().sort_index()
        hit_from_rank=details.set_index(['pid','model'])['rank'].le(10).astype(int).sort_index()
        np.testing.assert_array_equal(hit_from_list.to_numpy(),hit_from_rank.to_numpy())
        queries=pd.read_csv(out/'test_queries.csv').set_index('pid')
        raw=pd.read_csv(Path(__file__).resolve().parent/'data'/'songs620.csv')
        catalogue=pd.read_csv(out/'catalogue.csv')
        uri_to_id=dict(zip(catalogue.track_uri,catalogue.item))
        for pid,q in queries.iterrows():
            ordered=raw[raw.pid==pid].sort_values('pos').track_uri.map(uri_to_id).to_numpy()
            np.testing.assert_array_equal(q[[f'h{j}' for j in range(5)]].to_numpy(),ordered[:5])
            self.assertEqual(q.target,ordered[5])


if __name__=='__main__':
    unittest.main()
