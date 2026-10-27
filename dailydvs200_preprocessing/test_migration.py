import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'ddp4_code'))
from migration import validate
class Tests(unittest.TestCase):
 def setUp(self):
  self.old=json.loads(Path('/remote-home/datasets/DailyDVS-200-prepared/gcstr_run1/config.json').read_text())
  self.new=copy.deepcopy(self.old);self.new.update(world_size=4,batch_norm='sync',workers=2)
 def test_allow_only_declared_changes(self):validate(self.old,self.new)
 def test_reject_training_and_data_changes(self):
  for key,value in [('batch_size',256),('lr',.001),('classes',50),('identity',{})]:
   new=copy.deepcopy(self.new);new[key]=value
   with self.assertRaises(ValueError):validate(self.old,new)
 def test_reject_representation_changes(self):
  self.new['code']['render_frames.py']='changed'
  with self.assertRaises(ValueError):validate(self.old,self.new)
if __name__=='__main__':unittest.main()
