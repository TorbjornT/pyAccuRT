import numpy as np
import unittest
from accuread import ReadART

class TestMaterialProfile(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('demo1',basefolder='accuread/tests/testdata',
            material_profile=True,runvarfile='sza.txt')
        self.MP = self.PA.material_profile


    def test_Nruns(self):
        self.assertEqual(2,len(self.MP))

    def test_atmo(self):
        for j in range(2):
            for i in range(14):
                self.assertTrue('Atmosphericgases' in self.MP[j][i])

    def test_water(self):
        for j in range(2):
            self.assertTrue('Purewater' in self.MP[j][14])
            self.assertTrue('WaterImpurityCCRR' in self.MP[j][14])
            
    def test_bottom(self):
        bz = [30000.0,50000.0,60000.0,70000.0,76000.0,80000.0,
              84000.0,88000.0,90000.0,92000.0,94000.0,96000.0,
              98000.0,100000.0,100100.0]
        for i in range(15):
            self.assertTrue(bz[i]==self.MP[0][i]['bottomdepth'])

    def test_ccrr(self):
        ccrr = self.MP[0][14]['WaterImpurityCCRR']
        self.assertTrue(ccrr['concentration'] == 1)
        self.assertTrue(ccrr['concentrationtype'] == 'sf')
        self.assertTrue(ccrr['opticaldepth'] == 12.54)
        self.assertTrue(ccrr['singlescatteringalbedo'] == 0.7231)
        self.assertTrue(ccrr['asymmetryfactor'] ==  0.8402)
        self.assertTrue(ccrr['absorptionoptdep'] == 3.474)
        self.assertTrue(ccrr['scatteringoptdep'] == 9.07)
        self.assertTrue(ccrr['absorption'] == 0.03474)
        self.assertTrue(ccrr['scattering'] == 0.0907)
        self.assertTrue(ccrr['deltafitscalingfactor'] == 0.2619)
