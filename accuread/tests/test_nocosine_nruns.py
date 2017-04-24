import numpy as np
import unittest
from accuread import ReadART

class TestAccuRead(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('nocosine',basefolder='accuread/tests/testdata',
            cosine=False,iops=True)

    def test_wl(self):
        wl = self.PA.wavelengths
        self.assertTrue(np.array_equal(np.array([400,500,600,700]),wl))



if __name__ == '__main__':
    unittest.main()