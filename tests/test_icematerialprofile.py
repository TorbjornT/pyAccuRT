import numpy as np
import unittest
from accuread import ReadART

class TestMaterialProfileWithIce(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('materialprofileice',basefolder='tests/testdata',
            material_profile=True)
        self.MP = self.PA.material_profile

    def test_ice(self):
        self.assertTrue('Icebrines' in self.MP[0][1])
        self.assertTrue('Icebrinesimpurity' in self.MP[0][1])
        self.assertTrue('Icebubbles' in self.MP[0][1])
        self.assertTrue('Pureice' in self.MP[0][1])

if __name__ == '__main__':
    unittest.main()