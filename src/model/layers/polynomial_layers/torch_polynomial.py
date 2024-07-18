import torch.nn as nn
import torch

class WeightedPolynomial(nn.Module):
    def __init__(self, degree: int):
        super(WeightedPolynomial, self).__init__()
        self.degree = degree
        self.weights = None

    def build(self, x):
        if len(x.size()) == 3:
            self.weights = nn.Parameter(torch.rand(x.shape[-2], x.shape[-1], self.degree + 1, dtype=torch.float))
        elif len(x.size()) == 4:
            self.weights = nn.Parameter(torch.rand(x.shape[-3], x.shape[-2], x.shape[-1], self.degree + 1, dtype=torch.float))
        else:
            self.weights = nn.Parameter(torch.rand(x.shape[-1], self.degree + 1, dtype=torch.float))
            
    @staticmethod
    def _hyperparameters():
        return {
            "degree":[1, 6, 1]
        }

    def forward(self, x):
        if self.weights is None:
            self.build(x)

        # Create powers of x up to the degree
        powers = torch.stack([x ** i for i in range(self.degree + 1)], dim=-1)
        # Multiply each power of x by the corresponding weight and sum them
        result = torch.sum(self.weights * powers, dim=-1)
        return result

class MultiLinear(nn.Module):
    def __init__(self, degree: int=5):
        super(MultiLinear, self).__init__()
        self.degree = degree
        self.module_list = None
        self.input_shape = None

    def build(self,x):
        self.input_shape = x.size(-1)
        self.module_list = torch.nn.ModuleList([torch.nn.Bilinear(self.input_shape,self.input_shape,self.input_shape) for _ in range(0,self.degree)])
    
    @staticmethod
    def _hyperparameters():
        return {
            "degree":[1, 6, 1]
        }
    
    def forward(self,x):
        if self.module_list is None:
            self.build(x)
        for bilinear in self.module_list:
            x = bilinear(x,x)
        return x

class LazyLinear(torch.nn.LazyLinear):

    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(out_features, bias, device, dtype)

    @staticmethod
    def _hyperparameters():
        return {
            "out_features":[20, 256]
        }
    
if __name__=="__main__":
    """
    img = torch.randn(64, 28, 32)
    print(WeightedPolynomial(3)(img).size())
    print(WeightedPolynomial(10)(img))
    img = torch.randn(64,64,28, 32)
    print(WeightedPolynomial(10)(img))
    img = torch.randn(64, 32)
    print(WeightedPolynomial(10)(img))
    """
    list_layers_encoder = []
    list_layers_decoder = []
    config = {"layer_1": (32, True),
              "layer_2": (16, True)}
    x = torch.randn(32, 48, 64)
    tensor = torch.randn(32, 48, 64)

    y = torch.nn.LazyLinear(32)(x)
    y = WeightedPolynomial(degree=5)(y)
    y = torch.nn.LazyLinear(16)(y)
    z = WeightedPolynomial(degree=5)(y)
    MultiLinear = MultiLinear(degree=5)
    z = MultiLinear(x)

