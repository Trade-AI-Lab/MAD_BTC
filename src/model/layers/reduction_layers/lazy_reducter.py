import torch

class LazyReducter(torch.nn.Module):
    def __init__(self, out_channels_1, out_channels_2,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_1 = torch.nn.LazyLinear(out_channels_1)
        self.layer_2 = torch.nn.LazyLinear(out_channels_2)

    def forward(self,x):
        x = self.layer_1(x)
        x = torch.transpose(x,-2,-1)
        x = self.layer_2(x)
        return torch.transpose(x,-2,-1)
    
    @staticmethod
    def _hyperparameters():
        return {
            "out_channels_1":[3,12],
            "out_channels_2":[3,12]
        }
if __name__=="__main__":
    tensor = torch.randn(64,32,28)
    print(LazyReducter(10,10)(tensor).size())