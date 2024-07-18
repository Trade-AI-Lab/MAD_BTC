import torch
import math

class LazyConv1d(torch.nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros', dilatation=1, groups=1,
                 device="cpu"):
        super(LazyConv1d, self).__init__()
        self.cnn = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilatation = dilatation
        self.groups = groups
        self.device = device
        self.in_channels = None
        self.lin = None
        
    @staticmethod
    def _hyperparameters():
        return {
            "out_channels":[32,128],
            "kernel_size":[1,4,1],
            "padding":[0,5,1],
        }
    def compute_lout(self):
        return int((self.lin + 2 * self.padding - self.dilatation * (self.kernel_size - 1) - 1) / self.stride + 1)

    def decreaser_dilatation(self):
        self.dilatation -= 1

    def increaser_padding(self):
        self.padding += 1

    def decreaser_kernel_size(self):
        self.kernel_size -= 1
        
    def decreaser_stride(self):
        self.stride -= 1

    def check_parameters(self):
        token = "dilatation"
        if self.compute_lout() <= 0:
            while self.compute_lout() <= 0:
                if token == "dilatation":
                    self.decreaser_dilatation()
                    token = "kernel_size"
                    pass
                elif token == "kernel_size":
                    self.decreaser_kernel_size()
                    token = "padding"
                    pass
                else:
                    self.increaser_padding()
                    token = "dilatation"
        self.dilatation = 1 if self.dilatation <=0 else self.dilatation

        while self.kernel_size > self.lin:
            self.decreaser_kernel_size()
        while self.dilatation > self.lin:
            self.decreaser_dilatation()

        if self.kernel_size<=0:
            self.kernel_size = 1
    

    def build(self, x):
        self.in_channels = x.size(-2)
        self.lin = x.size(-1)
        self.check_parameters()
        self.cnn = torch.nn.Conv1d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   dilation=self.dilatation,
                                   groups=self.groups,
                                   padding_mode=self.padding_mode,
                                   device=self.device)

    def get_parameters(self):
        parameters_dict = {
            'out_channels': self.in_channels,
            'prev_lin': self.lin,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'padding_mode': self.padding_mode,
            'dilatation': self.dilatation,
            'groups': self.groups,
            'device': self.device
        }
        return parameters_dict


    def forward(self, x):
        if self.cnn is None:
            self.build(x)
        return self.cnn(x)


class LazyConv2d(torch.nn.Module):
    def __init__(self, out_channels, kernel_size=[3, 3], stride=[1, 1], padding=[0, 0], padding_mode='zeros',
                 dilatation=[1, 1], groups=1, device="cpu"):
        super(LazyConv2d, self).__init__()
        self.cnn = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilatation = dilatation
        self.groups = groups
        self.device = device
        self.in_channels = None
        self.hin = None
        self.win = None

    def compute_hout(self):
        return int(
            (self.hin + 2 * self.padding[0] - self.dilatation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)

    def compute_wout(self):
        return int(
            (self.win + 2 * self.padding[1] - self.dilatation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

    def decreaser_dilatation_0(self):
        self.dilatation[0] -= 1

    def increaser_padding_0(self):
        self.padding[0] += 1

    def decreaser_kernel_size_0(self):
        self.kernel_size[0] -= 1

    def decreaser_dilatation_1(self):
        self.dilatation[1] -= 1

    def increaser_padding_1(self):
        self.padding[1] += 1

    def decreaser_kernel_size_1(self):
        self.kernel_size[1] -= 1

    def check_parameters_hout(self):
        token = "dilatation"
        if self.compute_hout() <= 0:
            while self.compute_hout() <= 0:
                if token == "dilatation":
                    self.decreaser_dilatation_0()
                    token = "kernel_size"
                    pass
                elif token == "kernel_size":
                    self.decreaser_kernel_size_0()
                    token = "padding"
                    pass
                else:
                    self.increaser_padding_0()
                    token = "dilatation"
        self.dilatation[0] = 1 if self.dilatation[0] <0 else self.dilatation[0]
        self.dilatation[1] = 1 if self.dilatation[1] <0 else self.dilatation[1]

    def check_parameters_wout(self):
        token = "dilatation"
        if self.compute_wout() <= 0:
            while self.compute_wout() <= 0:
                if token == "dilatation":
                    self.decreaser_dilatation_1()
                    token = "kernel_size"
                    pass
                elif token == "kernel_size":
                    self.decreaser_kernel_size_1()
                    token = "padding"
                    pass
                else:
                    self.increaser_padding_1()
                    token = "dilatation"

    def build(self, x):
        self.in_channels = x.size(-3)
        self.hin = x.size(-2)
        self.win = x.size(-1)
        self.check_parameters_hout()
        self.check_parameters_wout()
        self.cnn = torch.nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=tuple(self.kernel_size),
                                   stride=tuple(self.stride),
                                   padding=tuple(self.padding),
                                   dilation=tuple(self.dilatation),
                                   groups=self.groups,
                                   padding_mode=self.padding_mode,
                                   device=self.device)

    def get_parameters(self):
        parameters_dict = {
            'out_channels': self.in_channels,
            'prev_hin': self.hin,
            'prev_win': self.win,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'padding_mode': self.padding_mode,
            'dilatation': self.dilatation,
            'groups': self.groups,
            'device': self.device
        }
        return parameters_dict


    def forward(self, x):
        if self.cnn is None:
            self.build(x)
        return self.cnn(x)


class LazyConvTranspose1d(torch.nn.Module):
    def __init__(self, out_channels, prev_lin ,kernel_size, stride=1, padding=0, padding_mode='zeros', dilatation=1,
                 groups=1,
                 device="cpu", *args, **kwargs):
        super(LazyConvTranspose1d, self).__init__()
        self.cnn = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilatation = dilatation
        self.groups = groups
        self.device = device
        self.lin = None
        self.output_padding = 0
        self.prev_lin = prev_lin
        self.in_channels = None

    def compute_lout(self):
        return (self.lin - 1) * self.stride - 2 * self.padding + self.dilatation * (
                    self.kernel_size - 1) + self.output_padding + 1

    def increase_output_padding(self):
        self.output_padding += 1

    def check_parameters_output_padding(self):
        while (self.compute_lout() - self.prev_lin) != 0:
            self.increase_output_padding()
    def build(self, x):
        self.lin = x.size(-1)
        self.in_channels = x.size(-2)
        self.check_parameters_output_padding()
        self.cnn = torch.nn.ConvTranspose1d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride,
                                            padding=self.padding,
                                            output_padding=self.output_padding,
                                            groups=self.groups,
                                            dilation=self.dilatation,
                                            padding_mode=self.padding_mode,
                                            device=self.device
                                            )
    def forward(self, x):
        if self.cnn is None:
            self.build(x)
        return self.cnn(x)

class LazyConvTranspose2d(torch.nn.Module):
    def __init__(self, out_channels, prev_hin, prev_win, kernel_size, stride=1, padding=0, padding_mode='zeros', dilatation=1,
                 groups=1,
                 device="cpu", *args, **kwargs):
        super(LazyConvTranspose2d, self).__init__()
        self.cnn = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilatation = dilatation
        self.groups = groups
        self.device = device

        self.hin = None
        self.win = None
        self.output_padding_0 = 0
        self.output_padding_1 = 0
        self.output_padding = None
        self.prev_hin = prev_hin
        self.prev_win = prev_win
        self.in_channels = None

    def compute_hout(self):
        return(self.hin -1)*self.stride[0]-2*self.padding[0] +self.dilatation[0]*(self.kernel_size[0] - 1) + self.output_padding_0 + 1

    def compute_wout(self):
        return (self.win - 1) * self.stride[1] - 2 * self.padding[1] + self.dilatation[1] * (
                    self.kernel_size[1] - 1) + self.output_padding_1 + 1

    def increase_output_padding_0(self):
        self.output_padding_0 += 1
    def increase_output_padding_1(self):
        self.output_padding_1 += 1

    def check_parameters_output_padding_0(self):
        while (self.compute_hout() - self.prev_hin) != 0:
            self.increase_output_padding_0()
    def check_parameters_output_padding_1(self):
        while (self.compute_wout() - self.prev_win) != 0:
            self.increase_output_padding_1()
    def build(self, x):
        self.win = x.size(-1)
        self.hin = x.size(-2)
        self.in_channels = x.size(-3)
        self.check_parameters_output_padding_0()
        self.check_parameters_output_padding_1()
        self.output_padding = [self.output_padding_0,self.output_padding_1]
        self.cnn = torch.nn.ConvTranspose2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=tuple(self.kernel_size),
                                   stride=tuple(self.stride),
                                   padding=tuple(self.padding),
                                   dilation=tuple(self.dilatation),
                                   groups=self.groups,
                                   padding_mode=self.padding_mode,
                                   output_padding=tuple(self.output_padding),
                                   device=self.device)

    def forward(self, x):
        if self.cnn is None:
            self.build(x)
        return self.cnn(x)
    
class PoolingLayer1d(torch.nn.Module):
    def __init__(self, norm_type:int,kernel_size:int,stride:int,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooling_layer = None
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def compute_lout(self):
        return math.floor((self.lin - self.kernel_size)/self.stride +1)
    
    def decrease_kernel_size(self):
        self.kernel_size -=1

    def check_parameters_output_lout(self):
        while self.compute_lout()<=0:
            self.decrease_kernel_size()

    def build(self,x:torch.tensor):
        self.lin = x.size(-1)
        self.check_parameters_output_lout()
        if self.lin == 1 or self.kernel_size <= 0:
            self.pooling_layer = torch.nn.Identity()

        else:
            self.pooling_layer = torch.nn.LPPool1d(float(self.norm_type),self.kernel_size,self.stride)

    def forward(self,x):
        if self.pooling_layer is None:
            self.build(x)
        return self.pooling_layer(x)
    
    @staticmethod
    def _hyperparameters():
        return {
            "norm_type":["1","2","4","6","inf"],
            "kernel_size": [1, 3, 1],
            "stride" : [1, 3, 1],
        }



if __name__ == "__main__":
    tensor = torch.randn((32, 64, 128))
    config = (32, 12, 4, 0, "zeros", 1, 1, 'cpu')
    lazy_cnn = LazyConv1d(*config)
    img = lazy_cnn(tensor)
    lazy_cnn_transpose1d = LazyConvTranspose1d(**lazy_cnn.get_parameters())
    output = lazy_cnn_transpose1d(img)
    print(output.size()==tensor.size())
    tensor = torch.randn((32, 64, 128, 64))
    lazy_cnn = LazyConv2d(32, [3, 3], [5, 2], [1, 1], dilatation=[3, 3])
    img = lazy_cnn(tensor)
    lazy_cnn_transpose2d = LazyConvTranspose2d(**lazy_cnn.get_parameters())
    output = lazy_cnn_transpose2d(img)
    print(output.size() == tensor.size())
    tensor = torch.randn((32, 64, 128))
    config = (42, 12, 4, 3, "zeros", 2, 1, 'cpu')
    lazy_cnn = LazyConv1d(*config)
    img = lazy_cnn(tensor)
    lazy_cnn_transpose1d = LazyConvTranspose1d(**lazy_cnn.get_parameters())
    output = lazy_cnn_transpose1d(img)
    print(output.size()==tensor.size())
    tensor = torch.randn((32, 64, 128))
    config = ("inf",4,4)
    lazy_cnn = PoolingLayer1d(*config)
    img = lazy_cnn(tensor)
    print(img.size())



    tensor = torch.randn((32, 64, 128))
    config = (32, 12, 4, 0, "zeros", 1, 1, 'cpu')
    module = LazyConv1d(*config)
    if not isinstance(module, torch.nn.Module) and module is not None:
        raise TypeError(f"{torch.typename(module)} is not a Module subclass")
    print(dir(module))
    hyperparameters = module._hyperparameters() if "_hyperparameters" in dir(module) else {}
    print(hyperparameters)
