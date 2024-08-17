
import numpy as np
from typing import Tuple , Literal
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# def convert_sec(seconds: float) -> str:
#     minutes, seconds = divmod(seconds, 60)
#     hours, minutes = divmod(minutes, 60)

#     return f'{int(hours)}h {int(minutes)} min {round(seconds,2)}s'


class PlottingTool():
    def __init__(self):
        self.plotting_component = True   
        
    def plotting_Single_ChaoticTrajectories (self, chaotic_trajs: np.ndarray) -> None:

        plt.figure(figsize=(25,4))

        for i , label in zip(range(3), ['x','y','z']):
            plt.subplot(1,3,i+1)
            plt.plot(chaotic_trajs[:,i] ,label = f'true_{label}')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)

        return plt.show()



    def plotting_Double_ChaoticTrajectories (self, chaotic_trajs_1: np.ndarray , 
                                            chaotic_trajs_2: np.ndarray , true_first:bool) -> None:

        if true_first == True:
            plt.figure(figsize=(25,4))
            for i , label in zip(range(3), ['x','y','z']):
                plt.subplot(1,3,i+1)
                plt.plot(chaotic_trajs_1[:,i] ,label = f'true_{label}')
                plt.plot(chaotic_trajs_2[:,i] ,label = f'pred_{label}')
                # plt.legend(loc='upper right')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)

        else:
            
            plt.figure(figsize=(25,4))
            for i , label in zip(range(3), ['x','y','z']):
                plt.subplot(1,3,i+1)
                plt.plot(chaotic_trajs_1[:,i] ,label = f'pred_{label}')
                plt.plot(chaotic_trajs_2[:,i] ,label = f'true_{label}')
                #plt.legend(loc='upper right')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)

        return plt.show()

class  DataCaster():

    def __init(self):
        self.tens_to_array = True
        self.array_to_tens =  True

    def tensor_to_numpy(self, tensor_data: torch.Tensor) -> np.ndarray:
        
        """
        Inputs:
                - the tensor data of shape (samples,features , 1 ) 
        
        Then, 
        convert the tensor data to a numpy array

        Outputs:
                - the numpy array data of shape (samples,features)
        """
        numpy_data = tensor_data.squeeze(2).detach().cpu().numpy()
        return numpy_data

    def numpy_to_tensor(self, numpy_data: np.ndarray , axis_to_extend =  Literal[1,2]) -> torch.tensor:
        
        """
        Inputs:
                - the numpy array data of shape (samples,features)
        
        Then, 
        convert the numpy array data to a tensor

        Outputs:
                - the tensor data of shape (samples,features ) 
        """
        tensor_data = torch.tensor(numpy_data, dtype =  torch.float32)

        if axis_to_extend == 1:
            tensor_data = tensor_data.unsqueeze(1)
        elif axis_to_extend == 2:
            tensor_data = tensor_data.unsqueeze(2)
        else:
            raise ValueError("axis_to_extend must be either '1'  or '2' ")
        return tensor_data





class LogCoshLoss(nn.MSELoss):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true , p =0.02):
        error = y_pred - y_true
        loss = torch.mean(torch.log(p*torch.cosh(error)))
        return loss