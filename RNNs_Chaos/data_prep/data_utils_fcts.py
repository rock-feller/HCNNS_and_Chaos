import torch
import numpy as np
import random
import pandas as pd
from typing import Tuple , Literal
from torch.utils.data import Dataset, DataLoader





class Normalization_Strategy():

    def __init__(self):

        self.data_normalization = True


    def Scale_ToNormalize(self, data :  torch.Tensor, 
                            scaling_factor: float) -> Tuple[np.ndarray , np.ndarray]:
        """
        Inputs:
                - the tensor data of shape (samples,features ) 
                - a scaling factor (float)
        
        Then, 
        Normalize the data  around 0 and multiply by the scaling factor according to the formula,
        scaled_data = scaling_factor*(data - data_averages)
        
        Outputs: 
                - the scaled data of shape  (samples,features,1) 
                - the average of the tensor data of shape (features,1)
        
                
        Inputs Shape: Tensor of shape (samples,features , 1) , float
        Outputs Shape: Tensor of shape(samples,features,1) , Tensor of shape (features,1)
        
        """

        tens_avgs = data.mean(axis=0)
        scaled_tens_data = scaling_factor*(data - tens_avgs) 

        
        return scaled_tens_data , tens_avgs

    def ScaleBackTo_originals(self, scaled_tens_data : torch.Tensor, scaling_factor: float ,
                           tens_avgs: torch.Tensor) ->  torch.Tensor:

        """This function takes as
        Inputs:
                - the scaled data of shape (samples,features,1) 
                - a scaling factor (float)
                - the average of the tensor data of shape (features,1)

        Then, 
        multiply the scaled data by the inverse of the scaling factor and
        add the averages along each axis
        
        
        Outputs: 
                - the original numpy array data of shape (samples,features)
                
        
        Inputs Shape: Tensor of shape (samples,features , 1) , float , Tensor of shape (features,1)
        Outputs Shape: Array of shape(samples,features,1) , Tensor of shape (samples , features)        
                
        """

        original_tens_data =  (scaled_tens_data / scaling_factor) + tens_avgs
        
        original_array_data  = original_tens_data.detach().cpu().numpy().squeeze()
        return original_array_data 




class Noisification_Strategy():
    def __init__(self ):
        self.add_noise = True


    def adding_sigmanoise_from_normal(self, chaos_trajs : torch.Tensor , sigma: float) -> torch.Tensor:
        """
        Inputs Shape: Tensor of shape (samples,features , 1) 
        Outputs Shape: Tensor of shape (samples,features , 1) 
        
        """
        noise_x =  np.random.normal(0,sigma,len(chaos_trajs))
        noise_y =  np.random.normal(0,sigma,len(chaos_trajs))
        noise_z =  np.random.normal(0,sigma,len(chaos_trajs))

        noisy_trajs = chaos_trajs + torch.tensor(np.vstack([noise_x,noise_y,noise_z]).T, dtype= torch.float32)


        return noisy_trajs.unsqueeze(2)
    

    def other_noise_strategy(self, chaos_trajs : torch.Tensor , sigma: float) -> torch.Tensor:
        return None
    




class SlidingWindowDataset(Dataset):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, tensor_data, window_size):
        """
        Initializes the dataset with sliding windows.
        
        Inputs shape:
          - tensor_data: tensor of shape (sam[ples , features)
          - window_size: number of time steps to look back
  
        """
        self.data = tensor_data
        self.window_size = window_size
        self.slided_data = self.sliding_windows_shift_to(tensor_data, window_size)

    def sliding_windows_shift_to(self, tensor_data, window_size) -> torch.Tensor:
        """
        Creates sliding window from data.
        
        Args:
        data (torch.tensor): Input tensor of shape (m, 1, p)
        window_size (int): The size of the sliding window.
        
        Returns:
        torch.tensor: 4D tensor of shape (num_batches, window_size, 1, p)
        """
        num_batches = len(tensor_data) - window_size + 1
        x = []
        for i in range(num_batches):
            _x = tensor_data[i:i+window_size]
            x.append(_x)
        return torch.stack(x).float().to(self.device)  # Ensures that the output is double precision

    def __len__(self):
        return len(self.slided_data)

    def __getitem__(self, idx):
        return self.slided_data[idx]
    


    def contextwindow_testdata_generator(self, context_size :int, fcast_size:int,  location: Literal["random_", "last_"]):

        if location.lower() == 'random_':
            """
             Inputs Shape:
              -  context_size : int
             -  fcast_size: int 
              - location : str : random_  or last_ 
              
              Outputs Shape:

               - context_window: tensor of shape (context_size, features)
               - windowdata_toforecast: tensor of shape (fcast_size, features)
              """

            full_data = self.sliding_windows_shift_to(self.data, context_size +fcast_size )
            random_window_id =  random.randint( 0, full_data.size(0))
            context_window   = full_data[random_window_id,:context_size].squeeze(1)
            windowdata_toforecast = full_data[random_window_id,context_size:].squeeze(1)
            
            return context_window  , windowdata_toforecast

        elif  location.lower() == 'last_':
            """Here you must only pass in the entire dataset as a tensor of shape [n_samples, n_features]"""
            full_data = self.sliding_windows_shift_to(self.data, context_size +fcast_size )
            context_window   = full_data[-1,:context_size].squeeze(1)
            windowdata_toforecast = full_data[-1,context_size:].squeeze(1)
            return context_window  , windowdata_toforecast

        else:
            raise ValueError("s0_nature must be either 'random_' or 'last_' ")
        

class SlidingWindowDataLoader(DataLoader):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #dtype = torch.float32
    def __init__(self, dataset, batch_size=32, shuffle=False, device=device):
        """
        Custom DataLoader that loads batches from a dataset.
        
        Args:
        dataset (SlidingWindowDataset): The dataset to load from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data every epoch.
        device (torch.device): The device to map the data to.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle )
        #self.device = device

    def __iter__(self):
        """
        Iterate over batches and map data to the specified device.
        """
        batches = super().__iter__()
        for batch in batches:
            yield batch.to(self.device)




class InpTarg_TSDataset(Dataset):
    def __init__(self, data: torch.Tensor, window_size: int):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int):
        input_seq = self.data[idx:idx+self.window_size]
        target_seq = self.data[idx+1:idx+self.window_size+1]
        return input_seq, target_seq

    @classmethod
    def create_train_loader(cls, data: torch.Tensor, window_size: int, batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        Inputs shape: 
            - tensor data of shaope (samples, features)
            - window_size: number of time steps to look back
            - batch_size: number of samples per batch
            - shuffle: whether to shuffle the data
        Outputs:
            - train_loader: DataLoader object : iterables over the dataset with inp_ and out_ batches of shape (batch_size, window_size, features)
        
        """
        train_dataset = cls(data, window_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader
    




import numpy as np
import torch



class data_splitter():
    def __init__(self):
        self.numpy_data_provided = True

    def create_sliding_windows(self, numpy_data, window_size):
        """
        Creates sliding windows from data.

        Args:
        numpy_data (np.ndarray): Input numpy array of shape (m, n).
        window_size (int): The size of the sliding window.

        Returns:
        torch.Tensor: 3D tensor of shape (num_windows, window_size, n).
        """
        num_windows = len(numpy_data) - window_size + 1
        windows = []
        
        for i in range(num_windows):
            window = numpy_data[i:i + window_size]
            windows.append(window)

        return np.array(windows)#.float()
    

    def ctextwindow_testdata_generator(self, numpy_data:np.ndarray, context_size :int, fcast_size:int,  location: Literal["random_", "last_"]):   
        import random

        if location.lower() == 'random_':
            """
                Inputs Shape:
                -  context_size : int
                -  fcast_size: int 
                - location : str : random_  or last_ 
                
                Outputs Shape:

                - context_window: tensor of shape (context_size, features)
                - windowdata_toforecast: tensor of shape (fcast_size, features)
                """

            full_data = self.create_sliding_windows( numpy_data , context_size +fcast_size )
            random_window_id =  random.randint( 0, numpy_data.shape[0])
            
            context_window_n_Tindex   = full_data[random_window_id,:context_size]#.squeeze(1)
            windowdata_toforecast_Tindex = full_data[random_window_id,-fcast_size:]#.squeeze(1)


            ctext_wdow_data_only =  torch.tensor(context_window_n_Tindex[:,1:].astype(np.float32)).squeeze(1)
            dateTindex_ctext_wdow = context_window_n_Tindex[:,0]

            wdowdata_tofcast_only = torch.tensor(windowdata_toforecast_Tindex[:,1:].astype(np.float32)).squeeze(1)
            dateTindex_fcast =  windowdata_toforecast_Tindex[:,0]
            
            return ctext_wdow_data_only  , wdowdata_tofcast_only, dateTindex_ctext_wdow , dateTindex_fcast
        

        elif  location.lower() == 'last_':

            """Here you must only pass in the entire dataset as a tensor of shape [n_samples, n_features]"""
            full_data = self.create_sliding_windows(numpy_data, context_size +fcast_size )

            context_window_n_Tindex   = full_data[-1,:context_size]#.squeeze(1)
            windowdata_toforecast_Tindex = full_data[-1,-fcast_size:]#.squeeze(1)

            ctext_wdow_data_only =  torch.tensor(context_window_n_Tindex[:,1:].astype(np.float32)).squeeze(1)
            dateTindex_ctext_wdow = context_window_n_Tindex[:,0]

            wdowdata_tofcast_only = torch.tensor(windowdata_toforecast_Tindex[:,1:].astype(np.float32)).squeeze(1)
            dateTindex_fcast =  windowdata_toforecast_Tindex[:,0]
            
            return ctext_wdow_data_only  , wdowdata_tofcast_only, dateTindex_ctext_wdow , dateTindex_fcast

        else:
            raise ValueError("s0_nature must be either 'random_' or 'last_' ")

class wind_data_transformer():
    def __init__(self):
        self.provide_dataframe = True


    ##  ===============================  Scaling  and Unscaling =================================

    def scale_around_zero( self, tensor_xy_both_heights: torch.Tensor)-> Tuple[torch.Tensor]:

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
        data = torch.tensor(ws_wd_only.to_numpy() , dtype=torch.float32)

        wv_20x = data[:, 0]* torch.cos(torch.deg2rad(data[:,1]))
        wv_20y = data[:, 0]* torch.sin(torch.deg2rad(data[:,1]))

        wv20_xy  =  torch.cat ( (wv_20x.unsqueeze(1) ,   wv_20y.unsqueeze(1)), dim=1)

        wv_60x = data[:, 2]* torch.cos(torch.deg2rad(data[:,3]))
        wv_60y = data[:, 2]* torch.sin(torch.deg2rad(data[:,3]))

        wv60_xy  =  torch.cat ( (wv_60x.unsqueeze(1) ,   wv_60y.unsqueeze(1)), dim=1)


        tensor_xy_both_heights = torch.cat ( (wv20_xy, wv60_xy) , dim=1 )
        
        return tensor_xy_both_heights


    def WindVector_toAngle(self, tensor_xy_one_height: torch.Tensor) -> torch.Tensor:

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
