import numpy as np
import unittest
from accuread import ReadART

class TestAccuRead(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('demo1',basefolder='accuread/tests/testdata',
            cosine=False,diffuse=True,
            runvarfile='sza.txt')

    def test_diffuse_size(self):
        irrshape = self.PA.diffuse_down.shape
        self.assertEqual((3,4,2),irrshape)

    def test_has_meta(self):
        self.assertTrue(hasattr(self.PA,'wavelengths'))
        self.assertTrue(hasattr(self.PA,'depths'))

    def test_wl(self):
        wl = self.PA.wavelengths
        self.assertTrue(np.array_equal(np.array([400,500,600,700]),wl))

if __name__ == '__main__':
    unittest.main()