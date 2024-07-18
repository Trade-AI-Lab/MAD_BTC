import torch.nn as nn
import torch

class LazyNorm1d(nn.Module):
    def __init__(self, norm1d_name:str,momentum:float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm1d_layer = getattr(torch.nn,norm1d_name)(momentum=momentum)
        self.norm1d_name = norm1d_name
        self.input_shape = None
        self.momentum = momentum
        
    @staticmethod
    def _hyperparameters():
        return {
            "norm1d_name": ["LazyInstanceNorm1d", "LazyBatchNorm1d"],
            "momentum": [0.05,0.1,0.15,0.2,0.25,0.3]
        }
    def build(self,x):
        self.input_shape = x.size()
        if self.input_shape[-1] == 1 and self.norm1d_name == "LazyInstanceNorm1d":
            self.norm1d_name = "LazyBatchNorm1d"
            self.norm1d_layer = getattr(torch.nn,self.norm1d_name)(momentum=self.momentum)
            
    def forward(self,x):
        if self.input_shape is None:
            self.build(x)
        return self.norm1d_layer(x)
    



if __name__=="__main__":
    tensor = torch.randn(64,128,64)
    lstm = LazyNorm1d(**{"norm1d_name":"LazyInstanceNorm1d","momentum":0.3})
    print(lstm(tensor))
    lstm = LazyNorm1d(**{"norm1d_name":"LazyBatchNorm1d","momentum":0.05})
    print(lstm(tensor))
    print(lstm(tensor).size())
    print(lstm.__class__.__name__)
    print(lstm._hyperparameters())
    tensor = torch.randn(64,5,1)
    lstm = LazyNorm1d(**{"norm1d_name":"LazyInstanceNorm1d","momentum":0.05})
    print(lstm(tensor))
    