import unittest
import torch
from lazy_convlayers import LazyConv1d, LazyConvTranspose1d, LazyConv2d, LazyConvTranspose2d

class TestLazyCNN(unittest.TestCase):

    def test_lazy_conv1d_transpose(self):
        tensor = torch.randn((32, 64, 128))
        lazy_cnn = LazyConv1d(32, 12, 4, 2, dilatation=32)
        img = lazy_cnn(tensor)
        lazy_cnn_transpose1d = LazyConvTranspose1d(*lazy_cnn.get_parameters())
        output = lazy_cnn_transpose1d(img)
        self.assertTrue(output.size() == tensor.size())

    def test_lazy_conv2d_transpose(self):
        tensor = torch.randn((32, 64, 128, 64))
        lazy_cnn = LazyConv2d(32, [3, 3], [5, 2], [1, 1], dilatation=[3, 3])
        img = lazy_cnn(tensor)
        lazy_cnn_transpose2d = LazyConvTranspose2d(*lazy_cnn.get_parameters())
        output = lazy_cnn_transpose2d(img)
        self.assertTrue(output.size() == tensor.size())

if __name__ == '__main__':
    unittest.main()
