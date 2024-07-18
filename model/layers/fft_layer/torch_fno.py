import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    def __init__(self,
                 out_channels,  # Number of output channels
                 modes1,        # Number of Fourier modes to multiply in the first dimension
                 modes2):       # Number of Fourier modes to multiply in the second dimension
        super(SpectralConv2d, self).__init__()

        self.in_channels = None
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self,x):
        self.in_channels = x.size(-3)
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.check_parameters()
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
    def decreaser_modes1(self):
        self.modes1 -= 1
    def decreaser_modes2(self):
        self.modes2 -= 1 
    def check_parameters(self):
        while self.in_channels<self.modes1:
             self.decreaser_modes1()
        while self.out_channels<self.modes2:
             self.decreaser_modes2()
    @staticmethod
    def _hyperparameters():
        return {
            'out_channels':[1,64],
            'modes1':[1,32],
            'modes2':[1,32],
        }
    def forward(self, x):
        if self.in_channels is None:
            self.build(x)
        batchsize = x.shape[0]
        #Compute Fourier coeffcients
        x_ft = torch.fft.fft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return torch.real(x)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)


class SpectralConv1d(torch.nn.Module):
    def __init__(self,
                 out_channels,
                 modes1):       # Number of Fourier modes to multiply in the first dimension
        super(SpectralConv1d, self).__init__()

        self.in_channels = None
        self.out_channels = out_channels
        self.modes1 = modes1

    def build(self,x):
        self.in_channels = x.size(-2)
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.check_parameters()
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, dtype=torch.cfloat))
        
    def decreaser_modes1(self):
        self.modes1 -= 1
    def check_parameters(self):
        while self.in_channels<self.modes1:
             self.decreaser_modes1()
    @staticmethod
    def _hyperparameters():
        return {
            'out_channels':[1,64],
            'modes1':[1,32]
        }
    def forward(self, x):
        if self.in_channels is None:
            self.build(x)
        batchsize = x.shape[0]
        #Compute Fourier coeffcients
        x_ft = torch.fft.fft(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = \
            self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        #Return to physical space
        x = torch.fft.ifft(out_ft,dim=-1)
        return torch.real(x)
    
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

if __name__=="__main__":
    img = torch.randn(64, 22, 28, 28)
    print(SpectralConv2d(14,4,8)(img).size())
    img = torch.randn(64,28,28)
    print(SpectralConv1d(100,100)(img).size())

