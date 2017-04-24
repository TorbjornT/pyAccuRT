import numpy as np
import unittest
from accuread import ReadART

class TestCalculatedVals(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('demo1',basefolder='accuread/tests/testdata',
            scalar=True,iops=True,direct=True,sine=True,radiance=True,
            runvarfile='sza.txt')

    def test_H_size(self):
        H = self.PA.calc_heatingrate()
        self.assertEqual((3,4,2),H.shape)


    def test_Kd_size(self):
        Kd = self.PA.diffuse_attenuation()
        self.assertEqual((2,4,2),Kd.shape)

    def test_Kdint_size(self):
        Kd = self.PA.diffuse_attenuation(integrated=True)
        self.assertEqual((2,2),Kd.shape)

if __name__ == '__main__':
    unittest.main()