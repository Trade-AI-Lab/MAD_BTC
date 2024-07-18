import torch.nn as nn
import torch

class LazyDropout(nn.Module):
    def __init__(self,p:float,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout_layer = getattr(torch.nn,"Dropout")(p)
        
    @staticmethod
    def _hyperparameters():
        return {
            "p":[0.1,0.2,0.3,0.4,0.5,0.6]
        }
            
    def forward(self,x):
        return self.dropout_layer(x)
    



if __name__=="__main__":
    tensor = torch.randn(64,128,64)
    lstm = LazyDropout(**{"dropout_name":"Dropout","p":0.3})
    print(lstm(tensor))
    print(lstm(tensor))
    print(lstm(tensor).size())
    print(lstm.__class__.__name__)
    print(lstm._hyperparameters())
    