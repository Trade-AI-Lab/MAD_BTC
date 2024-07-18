import torch
import torch.fft as fft

class Neural_Transform_FT(torch.nn.Module):
    def __init__(self):
        super(Neural_Transform_FT, self).__init__()
        self.input_shape = None
        self.fc_x = None
        self.fc_y = None
        self.img_3D = None

    def build(self,img):
        self.input_shape = img.size()
        if len(self.input_shape) > 3:
            self.img_3D = True
        last_size = self.input_shape[-1]
        self.fc_x = torch.nn.Sequential(
            torch.nn.Linear(last_size,64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, last_size)
        )
        self.fc_y = torch.nn.Sequential(
            torch.nn.Linear(last_size, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, last_size)
        )

    def forward(self, img):
        if self.input_shape is None:
            self.build(img)
        batch_size = img.size(0)
        X,Y = torch.meshgrid(torch.fft.fftfreq(self.input_shape[-2]),
                                        torch.fft.fftfreq(self.input_shape[-1]),
                                        indexing="ij")
        X = X.unsqueeze(0).repeat(batch_size, 1, 1)
        Y = Y.unsqueeze(0).repeat(batch_size, 1, 1)
        Z = (self.fc_x(X)*X + self.fc_y(Y)*Y).unsqueeze(1).repeat(1, self.input_shape[-3], 1, 1) if self.img_3D else self.fc_x(X)*X + self.fc_y(Y)*Y
        return torch.real(torch.fft.fft(Z*img))



if __name__=="__main__":
    import torch
    import torch.fft as fft
    import matplotlib.pyplot as plt

    input_data = torch.randn(64,10,256 ,256)
    model = Neural_Transform_FT()
    model(input_data)
    input_data = torch.randn(64,10,256)
    model = Neural_Transform_FT()
    model(input_data)
    # Save the model to a file
    # if __name__ == "__main__":
    #     import torch
    #     import torch.fft as fft
    #     import matplotlib.pyplot as plt
    #     from model.layers.fft_layer.torch_mellin_transform import *
    #
    #     input_data = torch.randn(16, 128, 256)
    #
    #     PATH = "model/layers/fft_layer/ntft_model.pth"
    #     model = torch.load(PATH)
    #     model(input_data)



