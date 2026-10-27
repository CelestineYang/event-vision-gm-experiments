"""Finish isolated environment setup; never starts training or GPU smoke."""
import os,sys,time,json,subprocess,shutil
from pathlib import Path
ROOT=Path('/remote-home/datasets/UCF101-feature-memory-runs')
WORK=Path('/remote-home/yanglansong/feature_memory')
BUNDLE=Path('/remote-home/datasets/Joind-dataset for Action Recognition/ucf101_feature_memory_migration_20261024_v1')
PYTHON=str(ROOT/'venv/bin/python')
status=ROOT/'environment_status.json'
def mark(stage,**kw):
 status.write_text(json.dumps(dict(stage=stage,**kw),indent=2));print(stage,flush=True)
def run(cmd,env=None):subprocess.run(cmd,check=True,cwd=WORK,env=env)
try:
 mark('waiting_for_torch_download')
 pid=int(sys.argv[1])
 while Path(f'/proc/{pid}').exists():time.sleep(10)
 run([PYTHON,'-c',"import torch,torchvision; assert torch.__version__=='2.0.1+cu118'; assert torchvision.__version__=='0.15.2+cu118'"])
 mark('installing_python_dependencies')
 run([PYTHON,'-m','pip','install','--no-cache-dir','setuptools==68.2.2','wheel==0.43.0','packaging==24.1','numpy==1.26.3','scipy==1.13.1','einops==0.7.0','transformers==4.35.2'])
 env=os.environ.copy();env.update(CUDA_HOME='/usr/local/cuda-11.8',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.6',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'),PIP_CACHE_DIR=str(ROOT/'pip_cache'))
 (ROOT/'tmp').mkdir(exist_ok=True)
 mark('installing_causal_conv1d')
 run([PYTHON,'-m','pip','install','causal-conv1d==1.4.0','--no-build-isolation','--no-deps'],env)
 build=ROOT/'mamba_build'
 shutil.copytree(BUNDLE/'environment/mamba_build',build)
 # Original build targets Ada sm89; local 3090 is Ampere sm86. Python/model code untouched.
 setup=build/'setup.py';text=setup.read_text();assert text.count('arch=compute_89,code=sm_89')==1
 setup.write_text(text.replace('arch=compute_89,code=sm_89','arch=compute_86,code=sm_86'))
 (ROOT/'build_adaptation.json').write_text(json.dumps(dict(file='mamba_build/setup.py',before='compute_89/sm_89',after='compute_86/sm_86',reason='RTX 3090 target',model_python_unchanged=True),indent=2))
 env['MAMBA_FORCE_BUILD']='TRUE'
 mark('building_custom_bimamba_sm86')
 run([PYTHON,'-m','pip','install',str(build),'--no-build-isolation','--no-deps'],env)
 mark('checking_installed_sources')
 check="""import pathlib,inspect,hashlib,mamba_ssm,causal_conv1d,torch,json
from mamba_ssm.modules.mamba_simple import Mamba
assert 'bimamba_type' in inspect.signature(Mamba.__init__).parameters
src=pathlib.Path('/remote-home/datasets/Joind-dataset for Action Recognition/ucf101_feature_memory_migration_20261024_v1/environment/mamba_build/mamba_ssm')
dst=pathlib.Path(mamba_ssm.__file__).parent
files=list(src.rglob('*.py'))
for f in files: assert f.read_bytes()==(dst/f.relative_to(src)).read_bytes(),str(f)
print('Verified installed Mamba Python sources:',len(files),flush=True)
"""
 run([PYTHON,'-c',check],env)
 mark('cpu_tests')
 run([PYTHON,'-m','unittest','test_gcstr_feature_memory','-v'],env)
 run([PYTHON,'prepare_feature_migration_config.py','--bundle',str(BUNDLE),'--output','configs/feature_memory_target'],env)
 mark('strict_real_checkpoint_load')
 run([PYTHON,'-c',"import json;from gcstr_feature_memory_fusion import load_model; c=json.load(open('configs/feature_memory_target/GM.json'));m=load_model(c);print('Real GM checkpoint strict CPU load passed')"],env)
 mark('ready_for_gpu_smoke',gpu_smoke_run=False,training_started=False,test_protocol='pending user decision')
except BaseException as exc:
 mark('failed',error=repr(exc));raise
