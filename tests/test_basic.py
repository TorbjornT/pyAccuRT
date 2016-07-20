import numpy as np
import unittest
from accuread import ReadART

class TestAccuRead(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('demo1',basefolder='tests/testdata',
            scalar=True,iops=True,direct=True,radiance=True,runvarfile='sza.txt')

    def test_wl(self):
        wl = self.PA.wavelengths
        self.assertTrue(np.array_equal(np.array([400,500,600,700]),wl))

    def test_depths(self):
        z = self.PA.depths
        self.assertTrue(np.array_equal(np.array([0,99999,100001]),z))

    def test_runvar(self):
        rv = self.PA.runvar
        self.assertTrue(np.array_equal(np.array([45,60]),rv))

    def test_cosdown_size(self):
        irrshape = self.PA.downdata.shape
        self.assertEqual((3,4,2),irrshape)

    def test_scldown_size(self):
        irrshape = self.PA.scalar_down.shape
        self.assertEqual((3,4,2),irrshape)

    def test_dirdown_size(self):
        irrshape = self.PA.direct_down.shape
        self.assertEqual((3,4,2),irrshape)

    def test_cosup_size(self):
        irrshape = self.PA.updata.shape
        self.assertEqual((3,4,2),irrshape)

    def test_sclup_size(self):
        irrshape = self.PA.scalar_up.shape
        self.assertEqual((3,4,2),irrshape)

    def test_dirup_size(self):
        irrshape = self.PA.direct_down.shape
        self.assertEqual((3,4,2),irrshape)

    def test_rad_size(self):
        radshape = self.PA.radiance.shape
        self.assertEqual((3,4,10,7,2),radshape)

    def test_polarangles(self):
        polang = self.PA.polarangles
        self.assertTrue(
            np.array_equal(
                np.array([179.92,160,140,120,100,80,60,40,20,0.081028]),
                polang))

    def test_azimuthanglesangles(self):
        azang = self.PA.azimuthangles
        self.assertTrue(
            np.array_equal(
                np.array([50,70,90,110,130,150,170]),
                azang))

if __name__ == '__main__':
    unittest.main()