import numpy as np
import unittest
from accuread.pyAccuRead import PyAccu

class TestAccuRead(unittest.TestCase):

    def setUp(self):
        self.PA = PyAccu('demo1',basefolder='testdata',
            scalar=True,iops=True,direct=True,runvarfile='sza.txt')

    def test_wl(self):
        wl = self.PA.wavelengths
        self.assertTrue(np.array_equal(np.array([400,500,600,700]),wl))

    def test_depths(self):
        z = self.PA.depths
        self.assertTrue(np.array_equal(np.array([0,99999,100001]),z))

    def test_runvar(self):
        rv = self.PA.runvar
        self.assertTrue(np.array_equal(np.array([45,60]),rv))

    def test_irr_size(self):
        irrshape = self.PA.downdata.shape
        self.assertEqual((3,4,2),irrshape)


if __name__ == '__main__':
    unittest.main()