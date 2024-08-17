import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from typing import Tuple , Literal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class ClimateDataProcessor():
    def __init__(self):
        self.provide_dataframe = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ##  ===============================  Scaling  and Unscaling =================================

    def scale_around_zero( self, tensor_xy_both_heights: torch.Tensor)-> Tuple[torch.Tensor]:
        tensor_xy_both_heights = tensor_xy_both_heights.to(self.device)
        """
        the input tensor should contain 4 columns:  wv20_x , wv20_y , wv60_x , wv60_y 

        ==================================================
        Input Shape: tensor of shape : (n_sample_wind vectors, 4)

        Output Shape: 
                - tensor of scaled data of shape : (n_sample_wind vectors, 4) 
                - tensor of averages along each of axis shape : (4 ,)
                - tensor of standard deviations along each of axis shape : (4,)
        ==================================================

         the out tensor  contain 4 scaled columns:  wv20_x , wv20_y , wv60_x , wv60_y 
         and the corresponding averages and standard deviations.
        
        """

        
        # ws_ws_only = df_wswd_both_heights.iloc[: , [1,3,4,5]]
        # data = torch.tensor(ws_ws_only.to_numpy() , dtype=torch.float32)
        scaling_factor = 0.2

        tens_avgs = tensor_xy_both_heights.mean(axis=0)#.unsqueeze(1)
        tens_stds =  tensor_xy_both_heights.std(axis=0)#.unsqueeze(1)
        scaled_tens_data = scaling_factor*((tensor_xy_both_heights - tens_avgs)/tens_stds)
        
        return scaled_tens_data , tens_avgs , tens_stds
    

    def unscale_back_around_zero(self, scaled_tens_data:torch.Tensor ,
                                 tens_avgs:torch.Tensor , tens_stds:torch.Tensor )-> np.ndarray:
        
        """
        
        The scaled_tens_data should have 4 columns in this order 
        wv20_x , wv20_y , wv60_x , wv60_y

    
        ==================================================
        Input Shape: 
        - tensor of scaled data of shape : (n_sample_wind vectors, 4) 
        - tensor of averages along each of axis shape : (4 ,)
        - tensor of standard deviations along each of axis shape : (4,)

        Ouput Shape:
        An nd array of shape: (n_sample_wind vectors, 4) 
        ==================================================

        the output tensor contain the unscaled ==> wv20_x , wv20_y ,wv60_x , wv60_y 
        
        """
          

        scaling_factor = 0.2

        original_tens_data =  (tens_stds*scaled_tens_data / scaling_factor) + tens_avgs
        
        #original_array_data  = original_tens_data.detach().cpu().numpy()#.squeeze()

        return original_tens_data 


    ##  =============================== =============================================



    ##  ===============================  Wind speed and Wind vector conversion and back =================================

    def wind_SpeedAngle_to_Vector(self, df_wswd_both_heights: pd.DataFrame ) -> torch.Tensor:


        """
        
        The dataframe should have 4 columns in this order 
        ws_20 , wd_20 , ws_60 , wd_60 
        the columns order should also match

    
        ==================================================
        Input Shape: dataframe of shape : (n_samples_ws_wd, 4)
        Output Shape: (n_samples_windvectors, 4)
        ==================================================

        the output tensor contain ==> wv20_x , wv20_y ,wv60_x , wv60_y 
        
        """
        
        ws_wd_only = df_wswd_both_heights.iloc[: ,1:5]
        data = torch.tensor(ws_wd_only.to_numpy() , dtype=torch.float32 , device =self.device)

        wv_20x = data[:, 0]* torch.cos(torch.deg2rad(data[:,1]))
        wv_20y = data[:, 0]* torch.sin(torch.deg2rad(data[:,1]))

        wv20_xy  =  torch.cat ( (wv_20x.unsqueeze(1) ,   wv_20y.unsqueeze(1)), dim=1)

        wv_60x = data[:, 2]* torch.cos(torch.deg2rad(data[:,3]))
        wv_60y = data[:, 2]* torch.sin(torch.deg2rad(data[:,3]))

        wv60_xy  =  torch.cat ( (wv_60x.unsqueeze(1) ,   wv_60y.unsqueeze(1)), dim=1)


        tensor_xy_both_heights = torch.cat ( (wv20_xy, wv60_xy) , dim=1 )
        
        return tensor_xy_both_heights


    def WindVector_toAngle(self, tensor_xy_one_height: torch.Tensor) -> torch.Tensor:
        tensor_xy_one_height = tensor_xy_one_height.to(self.device)
        """
        the input tensor should contain 2 colums:  wv_x , wv_y

        ==================================================
        Input Shape: tensor of shape : (n_sample_vectors, 2)
        Output Shape: (n_samples_wd, 1)
        ==================================================

        the output tensor contains 1 columns: wd_ for a given height
        """
        wx, wy = tensor_xy_one_height[:, 0], tensor_xy_one_height[:, 1]

        # Initialize phi tensor with zeros
        phi = torch.zeros_like(wx)

        # Condition 1: wx == 0 and wy == 0
        condition_1 = (wx == 0) & (wy == 0)
        phi[condition_1] = 0

        # Condition 2: wx == 0 and wy > 0
        condition_2 = (wx == 0) & (wy > 0)
        phi[condition_2] = np.pi / 2

        # Condition 3: wx == 0 and wy < 0
        condition_3 = (wx == 0) & (wy < 0)
        phi[condition_3] = 3 * np.pi / 2

        # Condition 4: wy == 0 and wx > 0
        condition_4 = (wy == 0) & (wx > 0)
        phi[condition_4] = 0

        # Condition 5: wy == 0 and wx < 0
        condition_5 = (wy == 0) & (wx < 0)
        phi[condition_5] = np.pi

        # Condition 6: wy > 0 and wx > 0
        condition_6 = (wy > 0) & (wx > 0)
        phi[condition_6] = torch.atan(wy[condition_6] / wx[condition_6])

        # Condition 7: wy > 0 and wx < 0
        condition_7 = (wy > 0) & (wx < 0)
        phi[condition_7] = np.pi + torch.atan(wy[condition_7] / wx[condition_7])

        # Condition 8: wy < 0 and wx > 0
        condition_8 = (wy < 0) & (wx > 0)
        phi[condition_8] = 2 * np.pi + torch.atan(wy[condition_8] / wx[condition_8])

        # Condition 9: wy < 0 and wx < 0
        condition_9 = (wy < 0) & (wx < 0)
        phi[condition_9] = np.pi + torch.atan(wy[condition_9] / wx[condition_9])

        return torch.rad2deg(phi.unsqueeze(1))
    # wind_vectors_tensor.shape

    def WindVector_toSpeed( self,  tensor_xy_both_heights: torch.Tensor) -> torch.Tensor:
        tensor_xy_both_heights = tensor_xy_both_heights.to(self.device)
        """
        the input tensor should contain 4 colums:  wv20_x , wv20_y , wv60_x , wv60_y 

        ==================================================
        Input Shape: tensor of shape : (n_sample_vectors, 4)
        Output Shape: (n_samples_ws_20_60, 2)
        ==================================================

        the output tensor  contain 2 colums:  ws_20 , ws_60
        
        """
        ws_20 = torch.sqrt((tensor_xy_both_heights[:, 0] ** 2) + (tensor_xy_both_heights[:, 1] ** 2))
        ws_60 = torch.sqrt((tensor_xy_both_heights[:, 2] ** 2) + (tensor_xy_both_heights[:, 3] ** 2))
        # tensors_ws_both_heights =   torch.cat ( (ws_20.unsqueeze(1) ,   ws_60.unsqueeze(1)) , dim=1)

        tensor_ws_20 = ws_20.unsqueeze(1)
        tensor_ws_60 = ws_60.unsqueeze(1)

        return tensor_ws_20 , tensor_ws_60

    ##  =============================== =============================================


    def transform_and_assemble(self, df_wswd_both_heights_and_Tindex: pd.DataFrame) -> np.ndarray:

        """
        Here the function takes the input dataframe , transform and assemble.

        the input dataframe should contain 7 columns in this order

        date_time, ws20 , wd20 , ws60 , wd60 ,t_index_cos, t_index_sin
        ==================================================
        Input Shape: dataframe  of shape : (n_samples, 7)

        Output Shape: 
            - numpy array of scaled and merged data of shape : (n_samples, 7) 
            - tensor of averages along each of axis shape : (4 , 1)
            - tensor of standard deviations along each of axis shape : (4 , 1)

        ==================================================
        the output numpy array contains the following information:

            - date_time (hh:mm)
            - wv20_x
            - wv20_y
            - wv60_x
            - wv60_y
            - t_index_cos
            - t_index_sin

        
        """

        tensor_xy_both_heights =  self.wind_SpeedAngle_to_Vector(df_wswd_both_heights_and_Tindex)#.detach().cpu().numpy()   

        tens_scaled_xy_both_heights , tens_avgs, tens_stds  = self.scale_around_zero(tensor_xy_both_heights)
        nd_array_date_time = pd.to_datetime(df_wswd_both_heights_and_Tindex.iloc[:,0]).dt.strftime('%H:%M').to_numpy().reshape(-1,1)

        nd_array_timeindex = df_wswd_both_heights_and_Tindex.iloc[:,-2:].to_numpy()

        np_array_merged = np.hstack((nd_array_date_time, tens_scaled_xy_both_heights.detach().cpu().numpy() , nd_array_timeindex))

        
        return  np_array_merged , tens_avgs , tens_stds
    
    def map_final_output_to_scaled_dframe ( self, df_wswd_both_heights_and_Tindex: pd.DataFrame ) -> pd.DataFrame: 

        np_array_merged , tens_avgs , tens_stds =  self.transform_and_assemble(df_wswd_both_heights_and_Tindex)

        df_scaled =  pd.DataFrame( np_array_merged , columns= ['date_time', 'wv20_x', 'wv20_y', 
                                                               'wv60_x', 'wv_60y', 't_index_cos', 't_index_sin'])
        
        return df_scaled
    
    #### ====================================  Transform  back  =======================================

    def transform_back_to_original(self, tens_scaled_xy_both_heights: torch.Tensor, tens_avgs: torch.Tensor, 
                                   tens_stds: torch.Tensor) -> np.ndarray:

        original_tensor_data = self.unscale_back_around_zero( tens_scaled_xy_both_heights ,
                                 tens_avgs , tens_stds )
        
        tensor_wd20 = self.WindVector_toAngle(original_tensor_data[:,:2])
        tensor_wd60 = self.WindVector_toAngle(original_tensor_data[:,2:])

        tensor_ws20 , tensor_ws60 = self.WindVector_toSpeed(original_tensor_data)
        
        merged_array_result =  torch.cat( (tensor_ws20, tensor_wd20,tensor_ws60,tensor_wd60 ), dim=1 ).detach().cpu().numpy()

        return merged_array_result

    
    def map_final_output_to_original_dframe ( self, tens_scaled_xy_both_heights: torch.Tensor, tens_avgs: torch.Tensor, 
                                   tens_stds: torch.Tensor , time_index_array: np.ndarray) -> pd.DataFrame: 

        merged_array_result = self.transform_back_to_original(tens_scaled_xy_both_heights,tens_avgs , tens_stds )

        np_array_merged = np.hstack(( np.expand_dims(time_index_array, axis=1), merged_array_result))
        

        df_scaled =  pd.DataFrame( np_array_merged , columns= ['date_time', 'WS20M', 'WD20M', 'WS60M', 'WD60M'])

        return df_scaled
    




            
            
            
            
            

