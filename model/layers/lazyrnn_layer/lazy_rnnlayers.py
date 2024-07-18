import torch.nn as nn
import torch

class LazyLstm(nn.Module):
    def __init__(self,out_channels=32, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False, proj_size=0, device=None):
        super(LazyLstm, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.device = device
        self.input_size = None

    def build(self,x):
        self.input_size = x.size(-1)
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.out_channels,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            proj_size=self.proj_size,
            device=self.device)
        
    def get_parameters(self):
        if self.bidirectional:
            self.input_size = self.input_size//2
        parameters_dict = {
        'out_channels': self.input_size,
        'num_layers': self.num_layers,
        'bias': self.bias,
        'batch_first': self.batch_first,
        'dropout': self.dropout,
        'bidirectional': self.bidirectional,
        'proj_size': self.proj_size,
        'device': self.device
        }
        return parameters_dict

    def forward(self,x):
        if self.input_size is None:
            self.build(x)
        return self.lstm(x)[0]
    
    @staticmethod
    def _hyperparameters():
        return {
            "out_channels": [12, 96],
            "dropout": [0.1,0.2,0.3,0.4,0.5,0.6],
            "num_layers":[1,4],
            "bidirectional" : [True,False],
        }



if __name__=="__main__":
    tensor = torch.randn(64,128,64)
    lstm = LazyLstm(**{"out_channels":128,"num_layers":3,"dropout":0.3,"bidirectional":True})
    print(lstm(tensor).size())
    print(lstm.__class__.__name__)
    print(lstm._hyperparameters())
    hyperparameters = lstm._hyperparameters()
    b = {key:hyperparameters[key] for key in hyperparameters}
    print(f"{LazyLstm}".split('.')[-1][:-2])
    print(LazyLstm._hyperparameters())
    superclass = LazyLstm.__bases__[-1]
    print("Immediate superclass:", superclass)