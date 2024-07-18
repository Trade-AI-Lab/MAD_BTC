import torch
import sys, os

sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).find("V_2.1_Claria_3_classes_AutoML_Pytorch")+len("V_2.1_Claria_3_classes_AutoML_Pytorch")])
from Model import nn
from Model.optuna_opt.utils_automl import LoopLazyLayers
APPLICATION_NAME = 'core_pl_opt'
import datetime
import inspect
def customPrint(to_print, flush=True):
    print(APPLICATION_NAME + " -" + str(os.getpid()) + "- " + str(datetime.datetime.now()) + " " + to_print,
          flush=flush)

class CoreArchitecture(torch.nn.Module):
    def __init__(self, trial, i:int, j:int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_config_list_layers = {
            "LazyConv1d":nn.LazyConv1d,
            "Activation":nn.Activation,
            "Dropout":nn.LazyDropout,
            "LazyNorm1d":nn.LazyNorm1d,
        }
        self.i = i
        self.j = j
        self.trial = trial
        self.cnn_loop = LoopLazyLayers(trial,i=self.i,j=self.j,config_list_layers=self.cnn_config_list_layers,boolean=True,num_layers_max=5) # network which outputs a list of torch tensor which are the ouputs of a cnn_config_list_layers
        self.i = self.cnn_loop.i
        self.j = self.cnn_loop.j
        self.architecture = torch.nn.ModuleDict({"loop_0_cnn":self.cnn_loop})
        choices = ["LazyLstm", "LazyLinear"]
        for k in range(1,self.cnn_loop.num_layers+1):
            layer = trial.suggest_categorical(f'{k}_layer_boolean', choices)
            config_list_layers = {
                layer:getattr(nn,layer),
                "Activation":nn.Activation,
                "Dropout":nn.LazyDropout,
                "LazyNorm1d":nn.LazyNorm1d,
            }
            loop_layer = LoopLazyLayers(self.trial,i=self.i,j=self.j+k,config_list_layers=config_list_layers,boolean=True,num_layers_max=3,list_ouput=False)
            self.architecture[f"loop_{k}_{layer}"] = loop_layer

        self.list_keys =  [key for key in self.architecture.keys() if key != "loop_0_cnn"]
        customPrint(f"CoreArchitecture{inspect.currentframe().f_code.co_name}")
        
    def forward(self,x):
        x , modulelist_output_cnn = self.architecture["loop_0_cnn"](x)
        list_output = [x] # add the last cnn's output in list_output
        k = 0
        for key in self.list_keys:
           list_output.append(self.architecture[key](modulelist_output_cnn[k]))
           k+=1
        return list_output
    

class ReducerArchitecture(torch.nn.Module):
    def __init__(self, trial, i:int, j:int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        
        self.config_list_layers = {
            "LazyConv1d":nn.LazyConv1d,
            "Activation":nn.Activation,
            "LazyNorm1d":nn.LazyNorm1d,
            "Dropout":nn.LazyDropout,
            "Linear":nn.LazyReducter,
            "Flatten":torch.nn.Flatten,
            "Activation":nn.Activation,

        }
        
        self.architecture = None
        self.last_architecture = None
        self.i = i
        self.j = j
    def build(self,list_input):
        self.architecture = torch.nn.ModuleDict({})
        for k in range(len(list_input)):
            loop_layer = LoopLazyLayers(self.trial,i=self.i,j=self.j+k,config_list_layers=self.config_list_layers,boolean=True,num_layers_max=1,list_ouput=False)
            self.architecture[f"reduction_layer_{k}"]=loop_layer

        self.last_architecture = torch.nn.Sequential(LoopLazyLayers(self.trial,i=self.i,j=self.j+k,config_list_layers=
                                                                    {
                                                                        "Linear":nn.LazyLinear,
                                                                        "Activation":nn.Activation,
                                                                        "Dropout":nn.LazyDropout,
                                                                    },boolean=True,num_layers_max=5,list_ouput=False))
    
    def forward(self,list_input):
        if self.architecture is None:
            self.build(list_input=list_input)
        x = torch.cat([self.architecture[key](list_input[int(key.split("_")[-1])])for key in self.architecture],dim=-1)
        return self.last_architecture(x)
        
class CryptoCurrencyArchitectureV1(torch.nn.Module):
    """
    Initialize the CryptoCurrencyArchitectureV1.

    This class defines a neural network architecture for cryptocurrency prediction tasks.
    It allows for the specification of multiple neural networks, each with its own core and reducer architectures.

    :param trial: A parameter indicating something about the trial
    :type trial: Whatever type trial is
    :param nb_class: Number of classes in the classification task
    :type nb_class: int
    :param num_nets: Number of neural networks
    :type num_nets: int
    """
    def __init__(self, trial, nb_class:int, num_nets:int, *args, **kwargs):
        super(CryptoCurrencyArchitectureV1, self).__init__(*args, **kwargs)
        self.trial = trial
        self.num_nets = num_nets

        self.core_architectures = torch.nn.ModuleDict({})
        self.reducer_architectures = torch.nn.ModuleDict({})

        for k in range(num_nets):
            core_architecture = CoreArchitecture(trial=self.trial, i=0, j=k)
            self.core_architectures[f'core_{k}'] = core_architecture
            print(core_architecture.i,k,"index")
            reducer_architecture = ReducerArchitecture(trial=self.trial, i=core_architecture.i, j=k)
            self.reducer_architectures[f'reducer_{k}'] = reducer_architecture

        customPrint(f"CryptoCurrencyArchitectureV1{inspect.currentframe().f_code.co_name}_last_architecture")
        self.last_architecture = torch.nn.Sequential(LoopLazyLayers(self.trial,i=max([self.reducer_architectures[key].i for key in self.reducer_architectures.keys()])+1,j=-1,config_list_layers=
                                                                    {
                                                                        "Linear":nn.LazyLinear,
                                                                        "Activation":nn.Activation,
                                                                        "Dropout":nn.LazyDropout,

                                                                    },boolean=True,num_layers_max=5,list_ouput=False),
                                                                    torch.nn.LazyLinear(nb_class),
                                                                    torch.nn.Softmax(dim=-1))
        customPrint(f"CryptoCurrencyArchitectureV1{inspect.currentframe().f_code.co_name}")


    def forward(self,inputs):
        """
        Perform forward pass through the network.

        :param inputs: Input data for each neural network
        :type inputs: Tuple of torch.Tensor
        :return: Output prediction
        :rtype: torch.Tensor
        """
        outputs = []
        for i in range(self.num_nets):
            core_output = self.core_architectures[f'core_{i}'](inputs[i])
            reducer_output = self.reducer_architectures[f'reducer_{i}'](core_output)
            outputs.append(reducer_output)
        x = torch.cat(outputs,dim=-1)
        return self.last_architecture(x)
        
