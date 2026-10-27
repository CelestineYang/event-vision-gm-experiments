"""Gesture GM initialization from independent pretrained branches, not UCF Fusion A."""
from pathlib import Path
import torch
from gesture_eventmamba_pretrained import PretrainedEventMambaV2
from gcstr_eventmamba_fusion import GCSTREventMambaFusion
from gcstr_feature_memory_fusion import FeatureMemoryFusion,sha256
from gcstr_secnet_fusion import VIEWS

class GestureWindowFusion(GCSTREventMambaFusion):
    def __init__(self,encoder):
        super().__init__(num_classes=10,point_encoder=encoder,freeze_images=True,point_aux=False)
    def forward_outputs(self,events,views):
        tokens=self.encode_views(views);features,coords=self.point_encoder.forward_features(events)
        return {'logits':self.fuse_features(features,coords,tokens),'point_logits':self.point_encoder.classify_features(features)}

class GestureFeatureMemoryFusion(FeatureMemoryFusion):
    def parameter_groups(self,config):
        groups,names=super().parameter_groups(config)
        lookup={g['name']:g for g in groups}
        # Fresh target classifier is optimized with the other new modules.
        fresh=[n for n in names['point'] if n.startswith('base.point_encoder.classifier.4.')]
        if fresh and 'new_modules' not in lookup:
            group=dict(params=[],lr=config['lr_new_modules'],name='new_modules')
            groups.append(group);lookup['new_modules']=group
        for name in fresh:
            index=names['point'].index(name)
            parameter=lookup['point']['params'].pop(index)
            names['point'].pop(index)
            lookup['new_modules']['params'].append(parameter)
            names['new_modules'].append(name)
        return groups,names

def load_model(config):
    if config.get('point_head_policy')!='reset_final_layer_for_official_actions_1_to_10':
        raise ValueError('Uploaded point classifier labels differ: explicit target head reset is required')
    if config['num_classes']!=10:raise ValueError('Gesture requires 10 classes')
    path=Path(config['point_checkpoint'])
    if sha256(path)!=config['point_sha256']:raise ValueError('Point checkpoint changed')
    base=GestureWindowFusion(PretrainedEventMambaV2.from_checkpoint(path))
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(config['seed']+41003).get_state())
        base.point_encoder.classifier[4].reset_parameters()
    base.point_encoder.load_report=dict(base.point_encoder.load_report,
        target_head_policy=config['point_head_policy'],
        discarded_source_head=True,target_labels='official actions 1..10; exclude other (11)',
        source_label_evidence='SpikingJelly 0.0.0.0.14 lexicographic folders and Train-only prediction audit')
    report={}
    for view in VIEWS:
        path=Path(config['image_checkpoints'][view])
        if sha256(path)!=config['image_sha256'][view]:raise ValueError('Image checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['view']!=view or saved['config']['classes']!=10:raise ValueError('Wrong Gesture image checkpoint')
        for name,value in saved['config']['identity'].items():
            if sha256(Path(config['store'])/name)!=value:raise ValueError('Image source data mismatch: '+name)
        base.image_encoders[view].load_state_dict(saved['model_state'],strict=True)
        report[view]={'path':str(path),'epoch':saved['epoch']}
    model=GestureFeatureMemoryFusion(base,config['mode'],config['seed'])
    model.source_config={'initialization':'independent pretrained branches; random fusion; zero residual gate/memory',
        'point':base.point_encoder.load_report,'images':report}
    return model
