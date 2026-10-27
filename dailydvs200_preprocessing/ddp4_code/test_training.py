import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from train_gcstr import Metrics, Windows, load_frozen_branch, GlobalBatchSampler


class TrainingTests(unittest.TestCase):
    def test_probability_vote(self):
        m=Metrics()
        m.update(torch.tensor([[3.,0.],[0.,1.],[0.,5.]]),torch.tensor([0,0,1]),['a','a','b'])
        metrics,rows=m.finish()
        self.assertEqual(metrics['videos'],2)
        self.assertEqual(metrics['windows'],3)
        self.assertEqual(metrics['video_top1'],1.)

    def test_global_batches_cover_windows_once(self):
        for size in (9, 25, 65, 67, 68, 7129):
            ranks=[list(GlobalBatchSampler(size,64,r,4,7)) for r in range(4)]
            self.assertEqual(len({len(b) for b in ranks}),1)
            flat=[i for batches in ranks for batch in batches for i in batch]
            self.assertEqual(sorted(flat),list(range(size)))
            self.assertTrue(all(batch for batches in ranks for batch in batches))

    def test_metric_merge_matches_single_rank(self):
        logits=torch.tensor([[3.,0.],[0.,1.],[0.,5.]])
        labels=torch.tensor([0,0,1]);ids=['a','a','b']
        whole=Metrics();whole.update(logits,labels,ids)
        merged=Metrics()
        for indices in ([0,2],[1],[]):
            part=Metrics()
            if indices:part.update(logits[indices],labels[indices],[ids[i] for i in indices])
            merged.merge(part)
        actual,rows=merged.finish();expected,expected_rows=whole.finish()
        for key in expected:self.assertAlmostEqual(actual[key],expected[key])
        self.assertEqual(rows,expected_rows)

    def test_cpu_training_resume_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=root/'store';store.mkdir()
            (store/'status.json').write_text(json.dumps(dict(state='complete',scope='smoke')))
            (store/'config.json').write_text('{}');(store/'samples.json').write_text('[]')
            for split in ('train','test'):
                rows=[]
                for i in range(2):
                    sample=f'{split}_{i}';name=sample+'.npy'
                    events=np.column_stack([np.arange(100)%346,np.arange(100)%260,np.arange(100)*9000,np.arange(100)%2]).astype('f4')
                    np.save(store/name,events)
                    rows.append(dict(valid=True,event_file=name,sample_id=sample,label=i,
                                     left=0,right=100,start_us=0,end_us=1000000,window_id=0))
                (store/(split+'_windows.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            ds=Windows(store,'train');ds.epoch=1
            first=ds[0][0];second=ds[0][0]
            for v in first:torch.testing.assert_close(first[v],second[v],rtol=0,atol=0)
            command=[sys.executable,str(Path(__file__).parent/'train_gcstr.py'),'--store',str(store),
                     '--output',str(root/'run'),'--device','cpu','--workers','0','--batch-size','2','--smoke','--test-as-val']
            subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            subprocess.run(command+['--resume'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            for view in ('cstr3','xt_p','ty_p'):
                model=load_frozen_branch(root/'run'/view/'best_loss.pt')
                self.assertFalse(model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertEqual(model.backbone.fc.out_features,50)
                rows=(root/'run'/view/'metrics.jsonl').read_text().splitlines()
                self.assertEqual(len(rows),1)


if __name__=='__main__':unittest.main()
