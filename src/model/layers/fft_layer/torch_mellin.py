import torch
import numpy as np

class Mellin(torch.nn.Module):
    def __init__(self):
        super(Mellin, self).__init__()
    def forward(self, img):
        return torch.sign(img)*torch.real(torch.fft.fft2(torch.log(torch.abs(img)+1e-6)))


if __name__ == '__main__':
    input_data = torch.randn(16, 256, 256)
    mellin_model = Mellin()
    output = mellin_model(input_data)
    # print(output.size())
    #
    # print(ImageTransformer(3, output_size=15)(input_data).size())

    print(output.size())