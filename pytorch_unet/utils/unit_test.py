import os
import sys
import unittest

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "pytorch_unet.trainer"

from pytorch_unet.utils.metrics import dice


class TestUNet(unittest.TestCase):
    """Test for the u_net script."""
    pass


class TestHyperparameter(unittest.TestCase):
    """Test for the hyperparameter script."""
    pass


class TestMultiGpu(unittest.TestCase):
    """Test for the multi gpu script."""
    pass


class TestPerformance(unittest.TestCase):
    """Test for the performance script."""
    pass


class TestAugmentations(unittest.TestCase):
    """Test for the augments script"""
    pass


class TestLoad(unittest.TestCase):
    """Test for the load script"""
    pass


class TestEvaluate(unittest.TestCase):
    """Test for the evaluate script"""
    pass


class TestInterpret(unittest.TestCase):
    """Test for the interpret script"""
    pass


class TestTrain(unittest.TestCase):
    """Test for the train script"""
    pass


class TestHelpers(unittest.TestCase):
    """Test for the helpers script"""
    pass


class TestMetrics(unittest.TestCase):
    """Test for the load script"""

    def test_dice(self):
        image1 = [0, 1, 0]
        image2 = [0, 1, 0]
        true_coeff = 1.0
        dice_coeff = dice(image1, image2)
        self.assertEqual(true_coeff, dice_coeff, msg="Dice Test failed")


class TestLogger(unittest.TestCase):
    """Test for the logger script"""
    pass


class TestPlot(unittest.TestCase):
    """Test for the plot script"""
    pass


if __name__ == '__main__':
    unittest.main()
