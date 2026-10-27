import unittest
import numpy as np
from prepare_dailydvs200 import bounds, sample_points

class ProtocolTests(unittest.TestCase):
    def test_half_open_and_final_event(self):
        t=np.array([0,499999,500000,999999,1000000,1499999,1500000])
        rows=list(bounds(t))
        self.assertEqual(rows[0],(0,1000000,0,4))
        seen=np.zeros(len(t),dtype=bool)
        for a,b,l,r in rows:
            seen[l:r]=True
            self.assertTrue(((t[l:r]>=a)&(t[l:r]<b)).all())
        self.assertTrue(seen.all())
        self.assertEqual(len(rows),len(set(x[0] for x in rows)))
    def test_short_and_exact_duration(self):
        self.assertEqual(list(bounds(np.array([0,10]))),[(0,1000000,0,2)])
        self.assertEqual(list(bounds(np.array([0,999999]))),[(0,1000000,0,2)])
        with self.assertRaises(ValueError):list(bounds(np.array([],dtype='i8')))
    def test_tail_and_gaps(self):
        for t in (np.array([0,2000000,3456789]),np.arange(0,3000000,300)):
            rows=list(bounds(t)); seen=np.zeros(len(t),bool)
            for _,_,l,r in rows:seen[l:r]=True
            self.assertTrue(seen.all())
            self.assertEqual(rows[-1][1],int(t[-1])+1)
    def test_sampling_coordinates_and_repeatability(self):
        e=np.array([[0,0,100,0],[319,239,200,1]])
        p=sample_points(e,100,1000100,7)
        np.testing.assert_array_equal(p,sample_points(e,100,1000100,7))
        self.assertEqual(p.shape,(2048,3))
        self.assertTrue(np.isfinite(p).all());self.assertTrue((np.diff(p[:,0])>=0).all())
        self.assertAlmostEqual(float(p[-1,0]),.0001,places=8)
        self.assertAlmostEqual(float(p[-1,1]),319/320,places=6)
        self.assertAlmostEqual(float(p[-1,2]),239/240,places=6)

if __name__=='__main__':unittest.main()
