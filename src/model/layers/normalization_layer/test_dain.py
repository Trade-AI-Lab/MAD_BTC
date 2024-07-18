import unittest
from dain import DainLayer
import torch
class TestDAINLayer(unittest.TestCase):

    def test_full_dain_layer(self):
        # Define your input data (e.g., a tensor)
        input_data = torch.randn(16, 32, 64)  # Replace with appropriate input size

        # Create an instance of the DAIN_Layer with 'full' mode
        full_dain_layer = DainLayer(mode="full",
                                     mode_log=False)


        # Pass the input through your layer
        output = full_dain_layer(input_data)

        # Add your assertions to validate the output
        self.assertEqual(output.shape, input_data.shape)  # Check if the output shape matches the input shape

    def test_adaptive_avg_dain_layer(self):
        input_data = torch.randn(16, 32, 64)  # Replace with appropriate input size
        # Create an instance of the DAIN_Layer with 'full' mode
        full_dain_layer = DainLayer(mode="full")

        # Add your assertions to validate the output
        self.assertEqual(full_dain_layer(input_data).shape, input_data.shape)  # Check if the output shape matches the input shape
        # Add more assertions as needed
    def test_double_dain_layer(self):
        input_data = torch.randn(16, 32, 64)  # Replace with appropriate input size
        # Create an instance of the dain_layer_both_axis with 'full' mode
        from dain import DainLayerBothAxis
        dain_layer_both_axis = DainLayerBothAxis(mode_log_2=False)

        # Add your assertions to validate the output
        self.assertEqual(dain_layer_both_axis(input_data).shape, input_data.shape)  # Check if the output shape matches the input shape
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
