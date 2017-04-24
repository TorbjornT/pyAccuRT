import numpy as np
import unittest
from accuread import ReadART

class TestAccuAlbedoTransmittance(unittest.TestCase):

    def setUp(self):
        self.PA = ReadART('demo1',basefolder='accuread/tests/testdata')
        self.albedo = self.PA.albedo(layer=1)
        self.integrated_albedo = self.PA.albedo(layer=1,integrated=True)
        self.transmittance = self.PA.transmittance(layers=(1,2))
        self.integrated_transmittance = \
            self.PA.transmittance(layers=(1,2),integrated=True)

    def test_albedo_shape(self):
        albshape = self.albedo.shape
        self.assertEqual((4,2),albshape)
        intalbshape = self.integrated_albedo.shape
        self.assertEqual((2,),intalbshape)

    def test_transmittance_shape(self):
        transmshape = self.transmittance.shape
        self.assertEqual((4,2),transmshape)
        inttransmshape = self.integrated_transmittance.shape
        self.assertEqual((2,),inttransmshape)

    def test_albedo_value(self):
        layer1_albedo = np.array([[ 0.05949973,  0.08332079],
                                  [ 0.05907679,  0.08921967],
                                  [ 0.04286238,  0.07483299],
                                  [ 0.03728944,  0.06954811]])
        layer1_int_albedo = np.array([ 0.05046739,  0.08058111])
        self.assertTrue(np.allclose(self.albedo,layer1_albedo,rtol=1e-6))
        self.assertTrue(np.allclose(self.integrated_albedo,
                                    layer1_int_albedo,rtol=1e-6))

    def test_transmittance_value(self):
        layers2_3_transmittance = np.array([[ 0.80166064,  0.77442729],
                                            [ 0.90366676,  0.87229914],
                                            [ 0.72643656,  0.68440757],
                                            [ 0.45346547,  0.40637683]])
        layers2_3_int_transmittance = np.array([ 0.76403963,  0.72655879])
        self.assertTrue(np.allclose(self.transmittance,
                                    layers2_3_transmittance,rtol=1e-6))
        self.assertTrue(np.allclose(self.integrated_transmittance,
                                    layers2_3_int_transmittance,rtol=1e-6))


if __name__ == '__main__':
    unittest.main()