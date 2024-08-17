
import torch
import torch.nn as nn

#from lorenz_rock import *
import torch.nn.functional as F
import time
from typing import Optional, Type , Literal , List , Tuple
from .HCNN_Cells import vanilla_cell , ptf_cell , lstm_cell , LargeSparse_cell
import torch.nn.utils.prune as prune

class Vanilla_ModelClimate(nn.Module):

    name =  "vanilla"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dtype = torch.float32
    def __init__( self, n_obs :int, n_hid_vars:int,n_ext_vars : int ,
                 s0_nature: Literal['zeros_', 'random_'] , train_s0:bool ):

        super (Vanilla_ModelClimate , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.n_ext_vars = n_ext_vars
        self.train_s0 = train_s0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # self.dtype = torch.float32
        self.s0_nature = s0_nature
        self.cell = vanilla_cell(n_obs ,n_hid_vars  , n_ext_vars )

        self.to(self.device)

    
    def initial_hidden_state(self , batch_size :int = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype , device=self.device), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype , device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype , device=self.device), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(batch_size, 1, self.n_hid_vars, dtype=self.dtype , device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")


  



    def forward(self, initial_state_vec , data_window : torch.Tensor   ) -> Tuple[torch.Tensor]:



        """ 
        If len data_window.shape == 3, then the input is a window of data, then
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1 , n_obs+n_ext_vars)

        If len data_window.shape == 4, then the input is a batch of windows of data, then, 
            Input Shape:
            - initial_state_vec:
            - data_window: (batch_size, seq_len, 1, n_obs+n_ext_vars)

        Output Shape:
            - Xions: (seq_len, 1, n_obs)
            - out_clust: (seq_len, 1, n_obs+n_ext_vars)
            - s_states: (seq_len, 1, n_hid_vars)
        
        """
        if len(data_window.shape) == 3: #(seq_len, 1, n_obs+n_ext_vars)

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape[0] , 1, self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec+ self.cell.B(data_window[0,:,-self.n_ext_vars:])


            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell(state = s_states[t] ,allow_transition=True , 
                                                                observation = data_window[t,:, :self.n_obs],
                                                                externals = data_window[t+1, :,-self.n_ext_vars:])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell(state =s_states[data_window.shape[0] -1] ,
                                                           allow_transition=False  , 
                                                           observation =  data_window[data_window.shape[0] -1, :, :self.n_obs])#,
                                                           #externals = data_window[data_window.shape[0] -1, :, -self.n_ext_vars:])






        elif len(data_window.shape) == 4: #(batch_size , seq_len, 1, n_obs+n_ext_vars)

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape[0],  data_window.shape[1],  1 , self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] = initial_state_vec + self.cell.B( data_window[:,0,:, -self.n_ext_vars:])
            #  self.initial_hidden_state().repeat(data_window.shape[0],1,1)



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell(state = s_states[:,t,:] ,
                                                                allow_transition=True ,
                                                                observation = data_window[:,t,:, :self.n_obs],
                                                                externals = data_window[:,t +1, :,-self.n_ext_vars:])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell(state = s_states[:,data_window.shape[1]-1,:] ,
                                                                                                   allow_transition=False  , 
                                                                                                   observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])
        # With allow_transition  = False. there is no need for an externals as it is only outut extraction that is performed
        return Xions , out_clust, s_states
    
    
    def forecast(self,  initial_state_vec , data_window: torch.Tensor , fcast_horizon:int, 
                  fut_externals: torch.Tensor) -> Tuple[torch.Tensor]:
        
        
        """ 
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1 , n_obs+n_ext_vars)
            - fcast_horizon: int
            - fut_externals: (fcast_horizon, 1, n_ext_vars)

        Output Shape:
            - f_Xions: (fcast_horizon, 1, n_obs)
            - fs_states: (fcast_horizon+1, 1, n_hid_vars)

        """

        with torch.no_grad():

            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            fs_states[0] = initial_state_vec  + self.cell.B(data_window[0,:,-self.n_ext_vars:])


            for past_t in range(data_window.shape[0] -1 ):

                f_Xions[past_t] , _, fs_states[past_t+1] = self.cell.forward(state = fs_states[past_t] ,allow_transition=True,
                                                            observation =  data_window[past_t,:, :self.n_obs],
                                                            externals = data_window[past_t+1 , :,-self.n_ext_vars:])

            
            f_Xions[past_t +1] , _, fs_states[past_t+2] = self.cell.forward(state = fs_states[past_t+1] ,allow_transition=True,
                                                            observation =  data_window[past_t +1 ,:, :self.n_obs],
                                                            externals = fut_externals[0,:,:]) 

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon - 1 ):

                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], 
                                                                                allow_transition=True , observation= None,
                                                                                externals = fut_externals[future_t +1  - data_window.shape[0]])


            f_Xions[future_t +1]  ,_, fs_states[future_t + 2] = self.cell.forward(state = fs_states[future_t +1], allow_transition=True,
                                                                                                                                  observation=None , externals= fut_externals[-1] )# 
                                                                                #externals = fut_externals[future_t +1  -data_window.shape[0]])
        # there will be a last external time point that will be used to construct the state at fcast_horizon +1 but not used. , the value of -1 is just a place holder
        return f_Xions,  fs_states


class PTF_ModelClimate(nn.Module):

    name =  "ptf"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32
    def __init__( self, n_obs :int, n_hid_vars:int,n_ext_vars:int,s0_nature: Literal['zeros_', 'random_'] ,
                 train_s0:bool , num_epochs:int , target_prob: float , drop_output:bool ):

        super (PTF_ModelClimate , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.n_ext_vars =n_ext_vars
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.num_epochs = num_epochs



        # self.prob = self.decrease_dropout_prob(num_epochs ,  current_epoch)   # Initialize prob as 0
        self.target_prob = target_prob
        # self.num_epochs = num_epochs
        # self.delta = self.target_prob / (self.num_epochs / 2)
        self.drop_output = drop_output
        # dtype = torch.float32
        self.cell = ptf_cell(n_obs=n_obs , n_hid_vars=n_hid_vars ,n_ext_vars=n_ext_vars  )# , self.drop_output )
        self.to(self.device)

    # def cell(self , prob):
    #     return ptf_cell(n_obs=self.n_obs , n_hid_vars=self.n_hid_vars , prob = prob)


    #     return ptf_cell(n_obs=self.n_obs , n_hid_vars=self.n_hid_vars , prob=0.)
    def initial_hidden_state(self , batch_size:int = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size, 1,self.n_hid_vars, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
                else:
                    raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")


    def decrease_dropout_prob(self,  current_epoch :int , prob:float  )-> float:

        """This function adjusts the dropout probability based on the current epoch
        Input shape:
        current_epoch : int

        the delta term that is added up to the current probability after each epoch 
        as from  num_epochs/2 is calculated as:

        delta = target_prob / (num_epochs / 2)

        
        Output shape:
        prob : float"""
        # self.target_prob = target_prob
        # self.num_epochs = num_epochs

        #delta = self.target_prob / (num_epochs / 2)


        if current_epoch >= (self.num_epochs / 2):
            new_prob = min(self.target_prob,  prob + (self.target_prob / (self.num_epochs / 2)))
            return new_prob

        else:
            new_prob = 0.
            return new_prob



    def forward(self, initial_state_vec, data_window : torch.Tensor  ,  prob:float) -> Tuple[torch.Tensor]:

        """ 
        If len data_window.shape == 3, then the input is a window of data, then
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1 , n_obs+n_ext_vars)
            - prob : float

        If len data_window.shape == 4, then the input is a batch of windows of data, then, 
            Input Shape:
            - initial_state_vec:
            - data_window: (batch_size, seq_len, 1, n_obs+n_ext_vars)

        Output Shape:
            - Xions: (seq_len, 1, n_obs)
            - out_clust: (seq_len, 1, n_obs+n_ext_vars)
            - s_states: (seq_len, 1, n_hid_vars)
        
        """


        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            #r_states = torch.zeros(data_window.shape[0], 1 , self.n_hid_vars,dtype=dtype, device=device)
            out_clust = torch.zeros(data_window.shape[0], 1, self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec+ self.cell.B(data_window[0,:,-self.n_ext_vars:])

            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell(state = s_states[t] ,prob = prob,
                                                                allow_transition=True , drop_output = True 
                                                                , observation = data_window[t,:, :self.n_obs],
                                                                externals = data_window[t+1, :,-self.n_ext_vars:])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell(state =s_states[data_window.shape[0] -1] ,prob= prob ,
                                                           allow_transition=False  ,drop_output = True, 
                                                           observation =  data_window[data_window.shape[0] -1, :,:self.n_obs])#,
                                                           #externals = data_window[data_window.shape[0]-1, :, -self.n_ext_vars:])


            #return out_clust, s_states   



        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            #r_states = torch.zeros(data_window.shape[0], data_window.shape[1], 1 , self.n_hid_vars,dtype=dtype, device=device)
            out_clust = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] = initial_state_vec+ self.cell.B( data_window[:,0,:, -self.n_ext_vars:])



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell.forward(state = s_states[:,t,:] ,prob = prob ,
                                                                allow_transition=True ,drop_output = True,
                                                                  observation = data_window[:,t,:, :self.n_obs],
                                                                  externals = data_window[:,t +1, :,-self.n_ext_vars:])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell(state = s_states[:,data_window.shape[1]-1,:] ,prob = prob,
                                            allow_transition=False ,drop_output = True,
                                              observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])
        # With allow_transition  = False. there is no need for an externals as it is only outut extraction that is performed
        return Xions , out_clust, s_states






    def forecast(self,  initial_state, data_window: torch.Tensor , 
                 fcast_horizon:int,fut_externals: torch.Tensor) -> Tuple[torch.Tensor]:
        
        """ 
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1, n_obs+n_ext_vars)
            - fcast_horizon: int
            - fut_externals: (fcast_horizon, 1, n_ext_vars)

        Output Shape:
            - f_Xions: (fcast_horizon, n_obs, 1)
            - fs_states: (fcast_horizon+1, 1, n_hid_vars)


        """
        with torch.no_grad():

            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            fs_states[0] = initial_state  + self.cell.B(data_window[0,:,-self.n_ext_vars:])


            for past_t in range(data_window.shape[0] -1 ):

                f_Xions[past_t] , _, fs_states[past_t+1] = self.cell.forward(state = fs_states[past_t] ,prob=0.  , allow_transition=True,drop_output=False,
                                                            observation =  data_window[past_t,:, :self.n_obs],
                                                            externals = data_window[past_t+1 , :,-self.n_ext_vars:])

            
            f_Xions[past_t +1] , _, fs_states[past_t+2] = self.cell.forward(state = fs_states[past_t+1] ,prob =0. , allow_transition=True,drop_output=False,
                                                            observation =  data_window[past_t +1 ,:, :self.n_obs],
                                                            externals = fut_externals[0,:,:]) 

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon - 1 ):

                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], prob=0.,
                                                                                allow_transition=True , drop_output=False, observation= None,
                                                                                externals = fut_externals[future_t +1  - data_window.shape[0]])


            f_Xions[future_t +1]  ,_, fs_states[future_t + 2] = self.cell.forward(state = fs_states[future_t +1], prob =0. , allow_transition=True,drop_output=False,
                                                                                                                                  observation=None , externals= fut_externals[-1] )# 
                                                                                #externals = fut_externals[future_t +1  -data_window.shape[0]])
        # there will be a last external time point that will be used to construct the state at fcast_horizon +1 but not used. , the value of -1 is just a place holder
        return f_Xions,  fs_states


    def cut_off_future_tsteps_hook(self, module, input, output):

        output = (output[0], output[1][:-1] , output[2][:-1])


        return output




class LSTM_FormulationClimate(nn.Module):

    name =  "lstm_form"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dtype = torch.float32
    
    def __init__( self, n_obs :int, n_hid_vars:int,n_ext_vars:int,
                 s0_nature: Literal['zeros_', 'random_'] , train_s0:bool ):

        super (LSTM_FormulationClimate , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.train_s0 = train_s0
        self.s0_nature= s0_nature
        self.n_ext_vars = n_ext_vars
        
        self.cell = lstm_cell(n_obs ,n_hid_vars , n_ext_vars )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # self.dtype = torch.float32


    def initial_hidden_state(self , batch_size:int = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size, 1,  self.n_hid_vars, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
                else:
                    raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")

        





    def forward(self, initial_state_vec,  data_window : torch.Tensor   ) -> Tuple[torch.Tensor]:


        """ 
        If len data_window.shape == 3, then the input is a window of data, then
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1 , n_obs+n_ext_vars)

        If len data_window.shape == 4, then the input is a batch of windows of data, then, 
            Input Shape:
            - initial_state_vec:
            - data_window: (batch_size, seq_len, 1, n_obs+n_ext_vars)

        Output Shape:
            - Xions: (seq_len, 1, n_obs)
            - out_clust: (seq_len, 1, n_obs+n_ext_vars)
            - s_states: (seq_len, 1, n_hid_vars)
        
        """

        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape[0], 1, self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec + self.cell.B(data_window[0,:,-self.n_ext_vars:])


            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell.forward(state = s_states[t] ,allow_transition=True , 
                                                                observation = data_window[t,:, :self.n_obs],
                                                                externals = data_window[t+1, :, -self.n_ext_vars:])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell.forward(state =s_states[data_window.shape[0] -1] ,
                                                           allow_transition=False  , 
                                                           observation =  data_window[data_window.shape[0] -1, :,:self.n_obs])
                                                           #externals = data_window[data_window.shape[0] -1, :, -self.n_ext_vars:])






        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] =  initial_state_vec + self.cell.B( data_window[:,0,:, -self.n_ext_vars:])



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell.forward(state = s_states[:,t,:] ,
                                                                allow_transition=True ,
                                                                observation = data_window[:,t,:, :self.n_obs],
                                                                externals = data_window[:,t+1,:, -self.n_ext_vars:])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell.forward(state = s_states[:,data_window.shape[1]-1,:] ,
                                                                                                   allow_transition=False  , 
                                                                                                   observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])#,
                                                                                                   #externals = data_window[:,data_window.shape[1]-1,:, -self.n_ext_vars:])
    # With allow_transition  = False. there is no need for an externals as it is only outut extraction that is performed
        return Xions , out_clust, s_states
    
    
    def forecast(self, initial_state,  data_window: torch.Tensor , 
                 fcast_horizon:int,fut_externals: torch.Tensor) -> Tuple[torch.Tensor]:

        """ 
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1, n_obs+n_ext_vars )
            - fcast_horizon: int
            - fut_externals: (fcast_horizon, 1, n_ext_vars )

        Output Shape:
            - f_Xions: (fcast_horizon, 1, n_obs)
            - fs_states: (fcast_horizon+1, 1, n_hid_vars)

        """
        with torch.no_grad():

            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            fs_states[0] = initial_state  + self.cell.B(data_window[0,:,-self.n_ext_vars:])


            for past_t in range(data_window.shape[0] -1 ):

                f_Xions[past_t] , _, fs_states[past_t+1] = self.cell.forward(state = fs_states[past_t] ,allow_transition=True,
                                                            observation =  data_window[past_t,:, :self.n_obs],
                                                            externals = data_window[past_t+1 , :,-self.n_ext_vars:])

            
            f_Xions[past_t +1] , _, fs_states[past_t+2] = self.cell.forward(state = fs_states[past_t+1] ,allow_transition=True,
                                                            observation =  data_window[past_t +1 ,:, :self.n_obs],
                                                            externals = fut_externals[0,:,:]) 

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon - 1 ):
 
                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], 
                                                                                allow_transition=True , observation= None,
                                                                                externals = fut_externals[future_t +1  - data_window.shape[0]])


            f_Xions[future_t +1]  ,_, fs_states[future_t + 2] = self.cell.forward(state = fs_states[future_t +1], allow_transition=True,
                                                                                                                                  observation=None , externals= fut_externals[-1] )# 
                                                                                #externals = fut_externals[future_t +1  -data_window.shape[0]])
        # there will be a last external time point that will be used to construct the state at fcast_horizon +1 but not used. , the value of -1 is just a place holder
        return f_Xions,  fs_states


class LargeSparse_ModelClimate(nn.Module):

    name =  "largesparse"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dtype = torch.float32
    def __init__( self, n_obs :int, n_hid_vars:int, n_ext_vars:int, pct_zeroed_weights: float, 
                 s0_nature: Literal['zeros_', 'random_'] , train_s0:bool ):

        super (LargeSparse_ModelClimate , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.n_ext_vars = n_ext_vars
        self.pct_zeroed_weights=pct_zeroed_weights
        self.train_s0 = train_s0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        #self.dtype = torch.float32
        self.s0_nature = s0_nature
        self.cell = LargeSparse_cell(n_obs ,n_hid_vars,n_ext_vars , pct_zeroed_weights )
        self.to(self.device)

    def initial_hidden_state(self , batch_size = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size,1,  self.n_hid_vars, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
                else:
                    raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")


    



    def forward(self, initial_state_vec, data_window : torch.Tensor   ) -> Tuple[torch.Tensor]:
        

        """ 
        If len data_window.shape == 3, then the input is a window of data, then
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1 , n_obs+n_ext_vars)

        If len data_window.shape == 4, then the input is a batch of windows of data, then, 
            Input Shape:
            - initial_state_vec:
            - data_window: (batch_size, seq_len, 1, n_obs+n_ext_vars)

        Output Shape:
            - Xions: (seq_len, 1, n_obs)
            - out_clust: (seq_len, 1, n_obs+n_ext_vars)
            - s_states: (seq_len, 1, n_hid_vars)
        
        """

        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape[0], 1, self.n_obs ,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec+ self.cell.B(data_window[0,:,-self.n_ext_vars:])

            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell(state = s_states[t] ,allow_transition=True , 
                                                                observation = data_window[t, :, :self.n_obs],
                                                                externals = data_window[t+1, :,-self.n_ext_vars:])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell(state =s_states[data_window.shape[0] -1] ,
                                                           allow_transition=False  , 
                                                           observation =  data_window[data_window.shape[0] -1, :,:self.n_obs],
                                                           externals = data_window[data_window.shape[0] -1, :, -self.n_ext_vars:])





        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_obs,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(out_clust.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] =initial_state_vec + self.cell.B( data_window[:,0,:, -self.n_ext_vars:])



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell(state = s_states[:,t,:] ,
                                                                allow_transition=True ,
                                                                observation = data_window[:,t,:, :self.n_obs],
                                                                externals = data_window[:,t+1,:, -self.n_ext_vars:])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell(state = s_states[:,data_window.shape[1]-1,:] ,
                                                                                                   allow_transition=False  , 
                                                                                                   observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs],
                                                                                                   externals = data_window[:,data_window.shape[1]-1,:, -self.n_ext_vars:])

        return Xions , out_clust, s_states
    
    
    def forecast(self,  initial_state ,  data_window: torch.Tensor , fcast_horizon:int  , fut_externals: torch.Tensor) -> Tuple[torch.Tensor , torch.Tensor]:
        
        """ 
        Input Shape: 
            - initial_state_vec:
            - data_window: (seq_len, 1, n_obs+n_ext_vars )
            - fcast_horizon: int
            - fut_externals: (fcast_horizon, 1, n_ext_vars )

        Output Shape:
            - f_Xions: (fcast_horizon, 1, n_obs)
            - fs_states: (fcast_horizon+1, 1, n_hid_vars)

        """

        with torch.no_grad():
        
            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            fs_states[0] = initial_state  + self.cell.B(data_window[0,:,-self.n_ext_vars:])


            for past_t in range(data_window.shape[0] -1 ):

                f_Xions[past_t] , _, fs_states[past_t+1] = self.cell.forward(state = fs_states[past_t] ,allow_transition=True,
                                                            observation =  data_window[past_t,:, :self.n_obs],
                                                            externals = data_window[past_t+1 , :,-self.n_ext_vars:])

            
            f_Xions[past_t +1] , _, fs_states[past_t+2] = self.cell.forward(state = fs_states[past_t+1] ,allow_transition=True,
                                                            observation =  data_window[past_t +1 ,:, :self.n_obs],
                                                            externals = fut_externals[0,:,:]) 

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon - 1 ):

                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], 
                                                                                allow_transition=True , observation= None,
                                                                                externals = fut_externals[future_t +1  - data_window.shape[0]])


            f_Xions[future_t +1]  ,_, fs_states[future_t + 2] = self.cell.forward(state = fs_states[future_t +1], allow_transition=True,
                                                                                                                                  observation=None , externals= fut_externals[-1] )# 
                                                                                #externals = fut_externals[future_t +1  -data_window.shape[0]])
        # there will be a last external time point that will be used to construct the state at fcast_horizon +1 but not used. , the value of -1 is just a place holder
        return f_Xions,  fs_states


