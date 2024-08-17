
import torch
import torch.nn as nn

#from lorenz_rock import *
import torch.nn.functional as F
import time
from typing import Optional, Type , Literal , List , Tuple
from .HCNN_Cells import vanilla_cell , ptf_cell , lstm_cell , LargeSparse_cell
import torch.nn.utils.prune as prune

class Vanilla_Model(nn.Module):

    name =  "vanilla"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    def __init__( self, n_obs :int, n_hid_vars:int,
                 s0_nature: Literal['zeros_', 'random_'] , train_s0:bool ):

        super (Vanilla_Model , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.train_s0 = train_s0
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.dtype = torch.float32
        self.s0_nature = s0_nature
        self.cell = vanilla_cell(n_obs ,n_hid_vars )

        self.to(self.device)

    
    def initial_hidden_state(self , batch_size = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size, 1, self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
                else:
                    raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")


  



    def forward(self, initial_state_vec , data_window : torch.Tensor   ) -> Tuple[torch.Tensor]:

        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec# self.initial_hidden_state()

            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell(state = s_states[t] ,allow_transition=True , 
                                                                observation = data_window[t, :self.n_obs])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell(state =s_states[data_window.shape[0] -1] ,
                                                           allow_transition=False  , 
                                                           observation =  data_window[data_window.shape[0] -1, :self.n_obs])





        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] = initial_state_vec#  self.initial_hidden_state().repeat(data_window.shape[0],1,1)



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell(state = s_states[:,t,:] ,
                                                                allow_transition=True ,
                                                                observation = data_window[:,t,:, :self.n_obs])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell(state = s_states[:,data_window.shape[1]-1,:] ,
                                                                                                   allow_transition=False  , 
                                                                                                   observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])

        return Xions , out_clust, s_states
    
    
    def forecast(self,  initial_state_vec , data_window: torch.Tensor , fcast_horizon:int) -> Tuple[torch.Tensor]:

        with torch.no_grad():

            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            fs_states[0] = initial_state_vec


            for t in range(data_window.shape[0]):

                f_Xions[t] , _, fs_states[t+1] = self.cell.forward(state = fs_states[t] ,allow_transition=True,
                                                            observation =  data_window[t, :self.n_obs])

                

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon):

                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], 
                                                                                allow_transition=True )

        return f_Xions,  fs_states


class PTF_Model(nn.Module):

    name =  "ptf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    def __init__( self, n_obs :int, n_hid_vars:int,s0_nature: Literal['zeros_', 'random_'] ,
                 train_s0:bool , num_epochs:int , target_prob: float , drop_output:bool ):

        super (PTF_Model , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.s0_nature = s0_nature
        self.train_s0 = train_s0
        self.num_epochs = num_epochs



        # self.prob = self.decrease_dropout_prob(num_epochs ,  current_epoch)   # Initialize prob as 0
        self.target_prob = target_prob
        # self.num_epochs = num_epochs
        # self.delta = self.target_prob / (self.num_epochs / 2)
        self.drop_output = drop_output
        # dtype = torch.float32
        self.cell = ptf_cell(n_obs=n_obs , n_hid_vars=n_hid_vars  )# , self.drop_output )
        self.to(self.device)

    # def cell(self , prob):
    #     return ptf_cell(n_obs=self.n_obs , n_hid_vars=self.n_hid_vars , prob = prob)


    #     return ptf_cell(n_obs=self.n_obs , n_hid_vars=self.n_hid_vars , prob=0.)
    def initial_hidden_state(self , batch_size = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size, 1,self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
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

        
        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            #r_states = torch.zeros(data_window.shape[0], 1 , self.n_hid_vars,dtype=dtype, device=device)
            out_clust = torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec# self.initial_hidden_state()

            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell(state = s_states[t] ,prob = prob,
                                                                allow_transition=True , drop_output = True , observation = data_window[t, :self.n_obs])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell(state =s_states[data_window.shape[0] -1] ,prob= prob ,
                                                           allow_transition=False  ,drop_output = True, observation =  data_window[data_window.shape[0] -1, :self.n_obs])


            #return out_clust, s_states


        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            #r_states = torch.zeros(data_window.shape[0], data_window.shape[1], 1 , self.n_hid_vars,dtype=dtype, device=device)
            out_clust = torch.zeros(data_window.shape,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] = initial_state_vec# self.initial_hidden_state().repeat(data_window.shape[0],1,1)



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell.forward(state = s_states[:,t,:] ,prob = prob ,
                                                                allow_transition=True ,drop_output = True, observation = data_window[:,t,:, :self.n_obs])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell(state = s_states[:,data_window.shape[1]-1,:] ,prob = prob,
                                            allow_transition=False ,drop_output = True, observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])

        return Xions , out_clust, s_states






    def forecast(self,  initial_state, data_window: torch.Tensor , fcast_horizon:int) -> tuple[torch.Tensor]:

        with torch.no_grad():

            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)

            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)


            fs_states[0] = initial_state

            for t in range(data_window.shape[0]):

                f_Xions[t] , _, fs_states[t+1] = self.cell.forward(state = fs_states[t] ,prob = 0., allow_transition=True,
                                                            drop_output = False , observation =  data_window[t, :self.n_obs])

                

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon):

                f_Xions[future_t]  ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t],prob = 0., 
                                                                                allow_transition=True , drop_output = False  )

        return f_Xions,  fs_states


    def cut_off_future_tsteps_hook(self, module, input, output):

        output = (output[0], output[1][:-1] , output[2][:-1])


        return output




class LSTM_Formulation(nn.Module):

    name =  "lstm_form"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    def __init__( self, n_obs :int, n_hid_vars:int,
                 s0_nature: Literal['zeros_', 'random_'] , train_s0:bool ):

        super (LSTM_Formulation , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.train_s0 = train_s0
        self.s0_nature= s0_nature
        
        self.cell = lstm_cell(n_obs ,n_hid_vars )
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.dtype = torch.float32


    def initial_hidden_state(self , batch_size = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size, 1,  self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
                else:
                    raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")

        





    def forward(self, initial_state_vec,  data_window : torch.Tensor   ) -> Tuple[torch.Tensor]:

        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec# self.initial_hidden_state()

            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell.forward(state = s_states[t] ,allow_transition=True , 
                                                                observation = data_window[t, :self.n_obs])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell.forward(state =s_states[data_window.shape[0] -1] ,
                                                           allow_transition=False  , 
                                                           observation =  data_window[data_window.shape[0] -1, :self.n_obs])





        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] =  initial_state_vec# self.initial_hidden_state().repeat(data_window.shape[0],1,1)



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell.forward(state = s_states[:,t,:] ,
                                                                allow_transition=True ,
                                                                observation = data_window[:,t,:, :self.n_obs])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell.forward(state = s_states[:,data_window.shape[1]-1,:] ,
                                                                                                   allow_transition=False  , 
                                                                                                   observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])

        return Xions , out_clust, s_states
    
    
    def forecast(self, initial_state,  data_window: torch.Tensor , fcast_horizon:int) -> Tuple[torch.Tensor]:

        with torch.no_grad():

            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            #fs_states[0] = self.state_s0
            fs_states[0] = initial_state


            for t in range(data_window.shape[0]):

                f_Xions[t] , _, fs_states[t+1] = self.cell.forward(state = fs_states[t] ,allow_transition=True,
                                                            observation =  data_window[t, :self.n_obs])

                

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon):

                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], 
                                                                                allow_transition=True )

        return f_Xions,  fs_states



class LargeSparse_Model(nn.Module):

    name =  "largesparse"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    def __init__( self, n_obs :int, n_hid_vars:int, prop_zeroed_weights: int, 
                 s0_nature: Literal['zeros_', 'random_'] , train_s0:bool ):

        super (LargeSparse_Model , self).__init__()
        self.n_hid_vars= n_hid_vars
        self.n_obs = n_obs
        self.prop_zeroed_weights=prop_zeroed_weights
        self.train_s0 = train_s0
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.dtype = torch.float32
        self.s0_nature = s0_nature
        self.cell = LargeSparse_cell(n_obs ,n_hid_vars,self.prop_zeroed_weights )
        self.to(self.device)

    def initial_hidden_state(self , batch_size = 1) -> torch.Tensor:


        if batch_size ==1:

            if self.s0_nature.lower() == "zeros_":

                return nn.Parameter(torch.zeros(1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)

            elif self.s0_nature.lower() == "random_":

                return nn.Parameter(torch.empty(1, self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            
            else:
                raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")
        else:
                
                if self.s0_nature.lower() == "zeros_":
    
                    return nn.Parameter(torch.zeros(batch_size, 1, self.n_hid_vars, dtype=self.dtype), requires_grad=self.train_s0)
    
                elif self.s0_nature.lower() == "random_":
    
                    return nn.Parameter(torch.empty(batch_size,1,  self.n_hid_vars, dtype=self.dtype).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
                
                else:
                    raise ValueError("s0_nature must be either 'zeros_' or 'random_' ")


    



    def forward(self, initial_state_vec, data_window : torch.Tensor   ) -> Tuple[torch.Tensor]:

        if len(data_window.shape) == 3:

            s_states = torch.zeros(data_window.shape[0], 1, self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)


            s_states[0] = initial_state_vec# self.initial_hidden_state()

            for t in range(data_window.shape[0] -1):

              Xions[t] , out_clust[t],s_states[t+1] = self.cell(state = s_states[t] ,allow_transition=True , 
                                                                observation = data_window[t, :self.n_obs])


            Xions[data_window.shape[0] -1] , out_clust[data_window.shape[0] -1] = self.cell(state =s_states[data_window.shape[0] -1] ,
                                                           allow_transition=False  , 
                                                           observation =  data_window[data_window.shape[0] -1, :self.n_obs])





        elif len(data_window.shape) == 4:

            s_states = torch.zeros(data_window.shape[0] ,  data_window.shape[1],  1 , self.n_hid_vars, dtype=self.dtype, device=self.device)
            out_clust = torch.zeros(data_window.shape,dtype=self.dtype, device=self.device)
            Xions =  torch.zeros(data_window.shape ,dtype=self.dtype, device=self.device)

            s_states[:,0,:] =initial_state_vec # self.initial_hidden_state().repeat(data_window.shape[0],1,1)



            for t in range(data_window.shape[1]-1):
                

                Xions[:,t,:] , out_clust[:,t,:],s_states[:,t+1,:] = self.cell(state = s_states[:,t,:] ,
                                                                allow_transition=True ,
                                                                observation = data_window[:,t,:, :self.n_obs])


            # if t == data_window.shape[1] -1:

            Xions[:,data_window.shape[1]-1,:]  , out_clust[:,data_window.shape[1]-1,:] = self.cell(state = s_states[:,data_window.shape[1]-1,:] ,
                                                                                                   allow_transition=False  , 
                                                                                                   observation = data_window[:,data_window.shape[1]-1,:, :self.n_obs])

        return Xions , out_clust, s_states
    
    
    def forecast(self,  initial_state ,  data_window: torch.Tensor , fcast_horizon:int) -> Tuple[torch.Tensor , torch.Tensor]:
        
        with torch.no_grad():
        
            fs_states = torch.zeros(data_window.shape[0]+ fcast_horizon +1,  1  , self.n_hid_vars, dtype=self.dtype, device=self.device)
            f_Xions = torch.zeros(data_window.shape[0]+ fcast_horizon , 1,  self.n_obs, dtype=self.dtype, device=self.device)
            fs_states[0] = initial_state 


            for t in range(data_window.shape[0]):

                f_Xions[t] , _, fs_states[t+1] = self.cell.forward(state = fs_states[t] ,allow_transition=True,
                                                            observation =  data_window[t, :self.n_obs])

                

            for future_t in range(data_window.shape[0] , data_window.shape[0]+ fcast_horizon):

                f_Xions[future_t] , _ ,fs_states[future_t+1] = self.cell.forward(state = fs_states[future_t], 
                                                                                allow_transition=True )

        return f_Xions,  fs_states

