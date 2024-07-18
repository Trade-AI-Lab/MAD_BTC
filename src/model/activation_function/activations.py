import torch

class Symlog(torch.nn.Module):
    """
        Applies the symlog. See https://arxiv.org/pdf/2301.04104v1.pdf
    """
    def forward(self,x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)
    
class Symexp(torch.nn.Module):
    """
        Applies the symexp. See https://arxiv.org/pdf/2301.04104v1.pdf
    """
    def forward(self, x):
        return torch.sign(x) * torch.exp(torch.abs(x)-1)
    
class T_alpha(torch.nn.Module):
    """
    T_alpha See https://www.sciencedirect.com/science/article/pii/S266665102300030X

    _extended_summary_

    :param torch: _description_
    :type torch: _type_
    """
    def __init__(self, activation_layer:torch.nn.Module=None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.sigmoid = torch.nn.Sigmoid()
        self.activation_layer = torch.nn.Identity() if activation_layer is None else activation_layer
    def forward(self,x):
        alpha = self.sigmoid(self.weight)
        x = self.activation_layer(x)
        return (1 - alpha) * torch.exp(-x) - alpha * x * (1 - alpha) * torch.exp(-x)



if __name__=="__main__":
    input = torch.randn((64,28,64))
    z = Symexp()(input)
    z = Symlog()(z)
    layer = T_alpha(Symlog())
    z = layer(z)
