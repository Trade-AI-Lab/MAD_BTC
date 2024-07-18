import torch
import optuna
import torch.nn as nn
import random

def OptunaListElements(name_layer:str,liste:list,key:str,trial:optuna.Trial):
    """
    Generates an optimal value for a hyperparameter using Optuna based on the given list.

    Args:
        name_layer (str): The name of the layer.
        liste (list): The list of values for the hyperparameter.
        key (str): The key representing the hyperparameter.
        trial (optuna.Trial): The Optuna trial object used for optimization.

    Returns:
        The suggested optimal value for the hyperparameter based on Optuna.

    Raises:
        None
    """
    type_element = type(liste[0])
    if type_element == bool or type_element == str:
        return trial.suggest_categorical(f"{name_layer}{key}", liste)
    else:
        if len(liste)==2:
            return getattr(trial,"suggest_"+type_element.__name__)(f"{name_layer}{key}", liste[0], liste[1], log=True)
        elif len(liste)==3:
             return getattr(trial,"suggest_"+type_element.__name__)(f"{name_layer}{key}", liste[0], liste[1], liste[2])
        elif len(liste)>3:
            return trial.suggest_categorical(f"{name_layer}{key}", liste)
    
def loop_initializer(layer:torch.nn.Module,i:float,j:float,trial:optuna.Trial)->dict:
    """
    Initializes hyperparameters for a layer by suggesting optimal values using Optuna.

    Args:
        layer (torch.nn.module): The layer for which hyperparameters are being initialized.
        trial (optuna.Trial): The Optuna trial object used for optimization.
        i (int): Index of the deep loop.
        j (int): Index of the width loop.

    Returns:
        List of suggested optimal values for the layer's hyperparameters.

    Raises:
        None
    """
    hyperparameters = layer._hyperparameters() if "_hyperparameters" in dir(layer) else {}
    name_layer = f"{layer}".split('.')[-1][:-2]
    return {key:OptunaListElements(f'{name_layer}_{key}_{i}_{j}',hyperparameters[key],key,trial) for key in hyperparameters} if hyperparameters  else {}

def check_nan(tensor:torch.Tensor,valuable_info=None):
    info = valuable_info
    contains_nan = torch.isnan(tensor).any().item()
    if contains_nan:
        raise "Tensor containing nan"

class LoopLazyLayers(nn.Module):
    def __init__(self, trial,i:int,j:int,config_list_layers:dict,boolean:bool=True,num_layers_max:int=5,list_ouput:bool=True,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.num_layers = trial.suggest_int(f'LoopLazyLayers_{i}_{j+random.randint(10, 2500)}',1,num_layers_max) if num_layers_max>1 else 1
        self.i = i
        self.j = j
        self.config_list_layers = config_list_layers
        self.module_dict_layers= torch.nn.ModuleDict({})
        self.boolean = boolean
        self.list_output = list_ouput
        for iterator in range(self.num_layers):
            self.loop_build(iterator)
 
    def loop_build(self,iterator)->tuple:
        for key in self.config_list_layers:
            layer_object = self.config_list_layers[key] # just a object name it is not a str 
            type_torchnn_boolean = True if "torch.nn" in str(type(layer_object)) else False
            key_name = key+f"_{iterator}"
            if (self.trial.suggest_categorical(f'bool_{key_name}',[True,False]) or self.boolean):
                self.module_dict_layers[key_name] = layer_object(**loop_initializer(layer_object,self.i+iterator,self.j,self.trial)) if not type_torchnn_boolean else layer_object
        
    def forward(self,x):
        if self.list_output:
            list_output = [x]
            for key in self.module_dict_layers:
                y = x
                x = self.module_dict_layers[key](x)
                if "Activation" in key:
                    list_output.append(x)
            if len(list_output) == 1:
                y = x
                list_output.append(x)
            return x , list_output
        else:
            for key in self.module_dict_layers:
                y = x
                x = self.module_dict_layers[key](x)
            return x 