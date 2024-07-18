import unittest
from tsmixer import MixerLayer
import torch
from layers.normalization_layer.dain import DainLayer
class TestMixerLayer(unittest.TestCase):

    def test_MixerLayer(self):
        # Define your input data (e.g., a tensor)
        input_data = torch.randn(16, 32, 64)  # Replace with appropriate input size

        # Create an instance of the DAIN_Layer with 'full' mode
        mixer_layer = MixerLayer(activation=torch.nn.ELU(), dropout=0.3,layernorm=None)

        # Pass the input through your layer
        output = mixer_layer(input_data)

        # Add your assertions to validate the output
        self.assertEqual(output.shape, input_data.shape)  # Check if the output shape matches the input shape

    def test_MixerLayer_DainLayer(self):
        input_data = torch.randn(16, 32, 64)  # Replace with appropriate input size
        # Create an instance of the DAIN_Layer with 'full' mode
        mixer_layer_dain_layer = MixerLayer(activation=torch.nn.ELU(),dropout=0.3,
                                            layernorm=DainLayer(mode="full"))

        # Add your assertions to validate the output
        self.assertEqual(mixer_layer_dain_layer(input_data).shape, input_data.shape)  # Check if the output shape matches the input shape
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
