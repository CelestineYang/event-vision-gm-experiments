"""Gesture CPU audit. CPU scan reference validates wiring, not CUDA kernel behavior."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from gesture_eventmamba_model import EventMamba
from gesture_eventmamba_pretrained import PretrainedEventMambaV2
from gesture_fusion import load_model
from gesture_sequence_dataset import GestureVideoSequenceDataset,point_input
from gcstr_video_sequence_dataset import keyed_rng
from train_gesture_gm import configuration,identity
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

ROOT=Path(__file__).resolve().parent


def cpu_inner(xz,cw,cb,xw,dw,A,B=None,C=None,D=None,delta_bias=None,delta_softplus=True,**kw):
    x,z=xz.chunk(2,dim=1);n,d,l=x.shape;rank=dw.shape[1];state=A.shape[-1]
    x=F.silu(F.conv1d(x,cw,cb,padding=cw.shape[-1]-1,groups=d)[...,:l])
    projected=F.linear(x.transpose(1,2),xw)
    delta=F.linear(projected[...,:rank],dw).transpose(1,2)
    B=projected[...,rank:rank+state].transpose(1,2) if B is None else B
    C=projected[...,-state:].transpose(1,2) if C is None else C
    return selective_scan_ref(x,delta,A,B,C,D,z,delta_bias,delta_softplus)

class GestureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config=configuration(ROOT/'configs/gesture_gm_smoke.json')

    def test_strict_checkpoint_and_forward_feature_parity(self):
        model=PretrainedEventMambaV2.from_checkpoint(self.config['point_checkpoint']).eval()
        state=torch.load(self.config['point_checkpoint'],map_location='cpu',weights_only=True)['net']
        self.assertEqual(set(state),set(model.state_dict()))
        for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
        ds=GestureVideoSequenceDataset(self.config,'train',1);points=ds.window(0)['events'][None]
        with torch.no_grad(),patch('mamba_ssm.modules.mamba_simple.mamba_inner_fn_no_out_proj',cpu_inner):
            torch.manual_seed(17);reference=EventMamba.forward(model,points)
            torch.manual_seed(17);features,coords=model.forward_features(points);actual=model.classify_features(features)
        torch.testing.assert_close(reference,actual,rtol=0,atol=0)
        self.assertEqual([tuple(f.shape) for f in features],[(1,512,48),(1,256,96),(1,128,192)])
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(all((c[:,1:,0]>=c[:,:-1,0]).all() for c in coords))

    def test_complete_splits_and_temporal_schedule(self):
        train=GestureVideoSequenceDataset(self.config,'train',1);test=GestureVideoSequenceDataset(self.config,'test')
        self.assertEqual((len(train.groups),train.windows),(1078,26816))
        self.assertEqual((len(test.groups),test.windows),(264,6959))
        self.assertFalse(set(train.groups)&set(test.groups))
        for key in ('filename','subject','event_file'):
            self.assertFalse({r[key] for r in train.rows}&{r[key] for r in test.rows})
        self.assertEqual({r['label'] for r in test.rows},set(range(10)))
        for ds in (train,test):
            seen=[]
            for chunk in ds.schedule:
                for j in range(chunk['start'],chunk['stop']):
                    seen.extend((v,j) for v in chunk['videos'] if j<len(ds.groups[v]))
            self.assertEqual(len(seen),ds.windows);self.assertEqual(len(set(seen)),ds.windows)
        smoke=GestureVideoSequenceDataset(self.config,'train',1,True)
        self.assertEqual(len(smoke.groups),3)
        self.assertTrue(any(not r['group_start'] for r in smoke.schedule))

    def test_shared_augmentation_and_sparse_test(self):
        ds=GestureVideoSequenceDataset(self.config,'train',1)
        import gesture_sequence_dataset as module
        captured={};sample=module.point_input;render=module.render
        def point(events,*args,**kwargs):
            captured['events']=tuple(a.copy() for a in events);return sample(events,*args,**kwargs)
        def frames(events,*args,**kwargs):
            t,x,y,p=captured['events'];np.testing.assert_array_equal(events,np.column_stack([x,y,t,p]))
            return render(events,*args,**kwargs)
        with patch.object(module,'point_input',point),patch.object(module,'render',frames):a=ds.window(0)
        b=ds.window(0)
        for k in ('events','cstr3','xt_p','ty_p'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        sparse=(np.array([0,50]),np.array([0,127]),np.array([0,127]),np.array([0,1]))
        pts,fallback=point_input(sparse,keyed_rng(0));self.assertEqual(fallback,0);self.assertEqual(pts.shape,(3,1024));self.assertTrue(np.isfinite(pts).all())
        test=GestureVideoSequenceDataset(self.config,'test')
        invalid=[i for i,r in enumerate(test.rows) if not r['valid']];self.assertEqual(len(invalid),0)
        for i in invalid:
            row=test.window(i);self.assertTrue(row['valid']);self.assertTrue(torch.isfinite(row['events']).all())

    def test_point_sampling_matches_preparation_and_renderer(self):
        ds=GestureVideoSequenceDataset(self.config,'test');r=ds.rows[0]
        a=np.load(ds.store/r['event_file'],mmap_mode='r')[r['left']:r['right']]
        rng=np.random.default_rng(33);indices=np.arange(len(a));rng.shuffle(indices);indices=np.sort(indices[:1024])
        expected=a[indices][:,[2,0,1]].astype(np.float32)
        expected[:,0]=(expected[:,0]-expected[:,0].min())/(expected[:,0].max()-expected[:,0].min()+1e-6)
        expected[:,1:]/=128
        actual,_=point_input((a[:,2],a[:,0],a[:,1],a[:,3]),np.random.default_rng(33))
        np.testing.assert_array_equal(expected.T,actual)
        self.assertEqual((ROOT/'thu_chl_render.py').read_bytes(),(ROOT.parent/'thu_preprocessing/render_frames.py').read_bytes())
        self.assertEqual(self.config['selection_metric'],'video_top1')

    def test_gm_initialization_freezing_and_identity(self):
        model=load_model(self.config);model.train()
        self.assertFalse(any(p.requires_grad for p in model.base.image_encoders.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.base.point_encoder.parameters()))
        self.assertTrue(all(not m.training for m in model.base.point_encoder.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)))
        groups,names=model.parameter_groups(self.config)
        state=torch.load(self.config['point_checkpoint'],map_location='cpu',weights_only=True)['net']
        for key,value in model.base.point_encoder.state_dict().items():
            if key.startswith('classifier.4.'):
                self.assertFalse(torch.equal(value,state[key]))
            else:torch.testing.assert_close(value,state[key],rtol=0,atol=0)
        self.assertIn('base.point_encoder.classifier.4.weight',names['new_modules'])
        self.assertNotIn('base.point_encoder.classifier.4.weight',names['point'])
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

        self.assertEqual({g['name'] for g in groups},{'point','existing_fusion','new_modules'})
        self.assertTrue(all(torch.count_nonzero(g.delta[-1].weight)==0 for g in model.extra))
        train=GestureVideoSequenceDataset(self.config,'train',1)
        record=identity(self.config,train,model);self.assertEqual(record['config']['num_classes'],10)
        (ROOT/'gesture_identity_audit.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':unittest.main()
