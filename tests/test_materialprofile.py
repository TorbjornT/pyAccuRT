import numpy as np
import unittest
from accuread import ReadART

class TestMaterialProfile(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('demo1',basefolder='tests/testdata',
            material_profile=True,runvarfile='sza.txt')
        self.MP = self.PA.material_profile


    def test_Nruns(self):
        self.assertEqual(2,len(MP))

    def test_atmo(self):
        for j in range(2):
            for i in range(14):
                self.assertTrue('Atmoshphericgases' in self.MP[j][i])

    def test_water(self):
        for j in range(2):
            self.assertTrue('Purewater' in self.MP[j][14])
            self.assertTrue('WaterImpurityCCRR' in self.MP[j][14])
            
    def test_bottom(self):
        bz = [3e5,5e5,6e5,7e5,7.6e5,8e5,8.4e5,8.8e5,9e5,
              9.2e5,9.4e5,9.6e5,9.8e5,1e6,1.001e6]
        for i in range(15):
            self.assertTrue(bz[i]==self.MP[0][i]['bottomdepth'])

    def test_ccrr(self):
        ccrr = self.MP[0][14]
        self.assertTrue(ccrr['concentration'] == 1)
        self.assertTrue(ccrr['concentrationtype'] == 'sf')
        self.assertTrue(ccrr['opticaldepth'] == 12.54
        self.assertTrue(ccrr['singlescatterinaglbedo'] == 0.7231
        self.assertTrue(ccrr['asymmetryfactor'] ==  0.8402
        self.assertTrue(ccrr['absorptionoptdep'] == 3.474
        self.assertTrue(ccrr['scatteringoptdep'] == 9.07
        self.assertTrue(ccrr['absorption'] == 0.03474
        self.assertTrue(ccrr['scattering'] == 0.0907
        self.assertTrue(ccrr['deltafitscalingfactor'] == 0.2619
