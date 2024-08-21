import torch
import torch.nn as nn
from typing import Optional, Type , Literal , List , Tuple


# Diagonal Matrix for HCNN with LSTM formulation
class DiagonalMatrix(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        if in_features != out_features:
            raise ValueError("DiagonalMatrix requires in_features to be equal to out_features for symmetry.")
        super(DiagonalMatrix, self).__init__(in_features, out_features, bias=bias)
        
        # Initialize the weight to be diagonal with ones
        nn.init.constant_(self.weight, 0)  # Set all weights to zero
        self.weight.data.fill_diagonal_(1)

        # Register hooks to adjust weights after gradient updates
        self.weight.register_hook(self._clamp_and_zero_out)

    def _clamp_and_zero_out(self, grad):
        """
        Hook to modify the gradients and weight matrix during the backward pass:
        - Zero out off-diagonal elements.
        - Clamp diagonal elements between 0 (exclusive) and 1 (inclusive).
        """
        with torch.no_grad():
            # Zero out off-diagonal elements
            mask = torch.eye(self.weight.shape[0], device=self.weight.device)
            self.weight.data *= mask

            # Clamp diagonal elements between 0 and 1
            self.weight.data = torch.clamp(self.weight.data, min=1e-6, max=1)  # min slightly greater than 0

        # Modify the gradient accordingly
        return grad * mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight, self.bias)
    


# Partial Teacher Forcing Dropout 
class partial_teacher_forcing(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super(partial_teacher_forcing, self).__init__(p, inplace)
        self.p = p
        self.inplace = inplace
    def forward( self , input: torch.Tensor ) -> torch.Tensor:

        #if inplace == True:

        if not self.training or self.p == 0 :
            return input
        else:
            scaled_output =  super().forward(input )
            return (1 - self.p) * scaled_output
        

class CustomLinear(nn.Linear):

    def __init__(self, in_features:int, out_features:int, bias=False):
        
        super(CustomLinear, self).__init__(in_features, out_features, bias=False)
        # Initialize weights and optionally biases uniformly within [-0.75, 0.75] and convert to float
        nn.init.uniform_(self.weight, -0.75, 0.75)
        self.weight = nn.Parameter(self.weight.float())  # Ensure weights are float
        if bias:
            nn.init.uniform_(self.bias, -0.75, 0.75)
            self.bias = nn.Parameter(self.bias.float())  # Ensure biases are float


class ptf_cell(nn.Module):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype =  torch.float32

    def __init__(self, n_obs: int, n_hid_vars: int , n_ext_vars : int):#, prob: float):#, drop_output: bool): #, s0_nature: str, train_s0: bool
        super(ptf_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.n_ext_vars = n_ext_vars
        # self.prob = prob
        #self.drop_output = drop_output
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameter initialization with specified dtype
        dtype = torch.float32  # Standard dtype for neural networks
        
        self.A = CustomLinear(in_features =n_hid_vars,out_features =n_hid_vars)
        self.B = CustomLinear(in_features =n_ext_vars ,out_features =n_hid_vars)  
        
        #self.ptf_dropout = nn.Dropout(self.prob)  # Use standard dropout
        
        #self.ptf_dropout = partial_teacher_forcing( p = self.prob)

        # Buffers
        self.register_buffer('ConMat', torch.eye(n_obs, n_hid_vars, dtype=dtype))
        self.register_buffer('Ide', torch.eye(n_hid_vars, dtype=dtype))

        self.to(self.device)

    def ptf_dropout(self, prob:float):
        return partial_teacher_forcing( p = prob)
    
    def forward(self, state: torch.Tensor,  prob:float, allow_transition :bool  ,drop_output : bool ,  
                externals : Optional[torch.Tensor] = None, observation: Optional[torch.Tensor] = None)-> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Input Shape:

            - state: (batch_size, 1, n_hid_vars)
            - prob: float
            - allow_transition: bool
            - externals: (batch_size, 1, n_ext_vars)
            - drop_output: bool
            - observation: (batch_size, 1, n_obs) or None
        
        Output Shape:

            If observation is not None:

                - If allow_transition is True:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)

                - If allow_transition is False:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)

            If observation is  None:

                    - expectation: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)


            - expectation: (batch_size, 1, n_obs)
            - output: (batch_size, 1, n_obs) if drop_output is False, else (batch_size, 1, n_hid_vars)
            - next_state: (batch_size, 1, n_hid_vars) if allow_transition is True, else None
        
        """

        expectation = torch.matmul(state, self.ConMat.T ) 

        if observation is not None:
            
            if not drop_output == True:
                
                output =  expectation - observation
                #output =  expectation 
                dropped_difference = self.ptf_dropout(prob)(output ).float()#, p=self.prob, training=self.training).float()
                
            else:
                

                dropped_difference = self.ptf_dropout(prob)(expectation - observation).float()#, p=prob, training=self.training).float()
                output = dropped_difference
                

            if allow_transition == True:
                
                teach_forc = torch.matmul(dropped_difference, self.ConMat )#output
                
                r_state = torch.matmul(state , self.Ide) - teach_forc

                next_state = self.A( torch.tanh(r_state)) + self.B( externals) 

                return expectation, output,  next_state
            
            else: 
                
                return expectation, output



        else:
            
            output = expectation
            r_state = torch.matmul(state , self.Ide)

            next_state = self.A( torch.tanh(r_state)) + self.B( externals) 

        return expectation, output, next_state 




class vanilla_cell(nn.Module):

    

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 
    def __init__(self, n_obs: int, n_hid_vars: int , n_ext_vars:int):
        super(vanilla_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.n_ext_vars = n_ext_vars
        #self.dtype = torch.float32 
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameter initialization 
         
        self.A = CustomLinear(in_features =n_hid_vars, out_features =n_hid_vars)
        self.B = CustomLinear(in_features =n_ext_vars ,out_features =n_hid_vars) 

        self.register_buffer('ConMat', torch.eye(n_obs, n_hid_vars, dtype= self.dtype))
        self.register_buffer('Ide', torch.eye(n_hid_vars, dtype=self.dtype))
        self.to(self.device)


    def forward(self, state: torch.Tensor, allow_transition :bool , externals: Optional[torch.Tensor] = None,
                 observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Input Shape:

            - state: (batch_size, 1, n_hid_vars)
            - allow_transition: bool
            - externals: (batch_size, 1, n_ext_vars)
            - drop_output: bool
            - observation: (batch_size, 1, n_obs) or None
        
        Output Shape:

            If observation is not None:

                - If allow_transition is True:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)

                - If allow_transition is False:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)

            If observation is  None:

                    - expectation: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)


            - expectation: (batch_size, 1, n_obs)
            - output: (batch_size, 1, n_obs) if drop_output is False, else (batch_size, 1, n_hid_vars)
            - next_state: (batch_size, 1, n_hid_vars) if allow_transition is True, else None
        
        """
        expectation = torch.matmul(state, self.ConMat.T ) 

        if observation is not None:
            
            
                
            output =  expectation - observation
        

            if allow_transition == True:
                
                teach_forc = torch.matmul(output, self.ConMat )#output
                
                r_state = torch.matmul(state , self.Ide) - teach_forc

                next_state = self.A( torch.tanh(r_state))+ self.B( externals)

                return expectation, output,  next_state
            
            else: 
                
                return expectation, output


        else:
            
            output = expectation

            r_state = torch.matmul(state , self.Ide)

        next_state = self.A( torch.tanh(r_state)) + self.B( externals)

        
        return expectation, output,  next_state 
    



class lstm_cell(nn.Module):

    

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 
    def __init__(self, n_obs: int, n_hid_vars: int , n_ext_vars:int):
        super(lstm_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars


        # Parameter initialization 
         
        self.A = CustomLinear(in_features =n_hid_vars, out_features =n_hid_vars)
        self.B = CustomLinear(in_features =n_ext_vars ,out_features =n_hid_vars)
        self.D =  DiagonalMatrix(in_features =n_hid_vars, out_features =n_hid_vars)
        self.register_buffer('ConMat', torch.eye(n_obs, n_hid_vars, dtype= self.dtype))
        self.register_buffer('Ide', torch.eye(n_hid_vars, dtype=self.dtype))
        self.to(self.device)


    def forward(self, state: torch.Tensor, allow_transition :bool , externals: Optional[torch.Tensor] = None,
                 observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        

        """
        Input Shape:

            - state: (batch_size, 1, n_hid_vars)
            - allow_transition: bool
            - externals: (batch_size, 1, n_ext_vars)
            - drop_output: bool
            - observation: (batch_size, 1, n_obs) or None
        
        Output Shape:

            If observation is not None:

                - If allow_transition is True:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)

                - If allow_transition is False:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)

            If observation is  None:

                    - expectation: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)


            - expectation: (batch_size, 1, n_obs)
            - output: (batch_size, 1, n_obs) if drop_output is False, else (batch_size, 1, n_hid_vars)
            - next_state: (batch_size, 1, n_hid_vars) if allow_transition is True, else None
        
        """
        expectation = torch.matmul(state, self.ConMat.T ) 

        if observation is not None:
            
            
                
            output =  expectation - observation
        

            if allow_transition == True:
                
                teach_forc = torch.matmul(output, self.ConMat )#output
                
                r_state = torch.matmul(state , self.Ide) - teach_forc

                lstm_block = self.A( torch.tanh(r_state)) -  r_state

                next_state =  r_state   +  self.D( lstm_block ) + self.B( externals)

                return expectation, output,  next_state
            
            else: 
                
                return expectation, output


        else:
            
            output = expectation

            r_state = torch.matmul(state , self.Ide)

        lstm_block = self.A( torch.tanh(r_state)) -  r_state
        
        next_state =  r_state   +  self.D(  lstm_block)  + self.B( externals)

        
        return expectation, output,  next_state 
    





class CustomSparseLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, pct_nonzero:float,  bias=False):
        super(CustomSparseLinear, self).__init__(in_features, out_features, bias=bias)
        
        # Ensure in_features equals out_features for a symmetric design
        if in_features != out_features:
            raise ValueError("in_features must be equal to out_features for this custom layer.")
        
        # Initialize weights uniformly within [-0.75, 0.75]
        nn.init.uniform_(self.weight, -0.75, 0.75)
        
        # Create a mask that will zero out 75% of the weights
        self.pct_nonzero = pct_nonzero
        total_elements = in_features * out_features
        non_zero_elements =  int(pct_nonzero* total_elements)#  int(0.25 * total_elements)  # 25% are non-zero
        mask = torch.zeros(total_elements, dtype=torch.bool)
        
        # Randomly pick indices to be non-zero
        non_zero_indices = torch.randperm(total_elements)[:non_zero_elements]
        mask[non_zero_indices] = True
        self.mask = mask.view(in_features, out_features)
        
        # Apply the mask to zero out 75% of the weights initially
        self.weight.data *= self.mask.float()

        # Register buffer for the mask to not consider it as a parameter
        self.register_buffer('update_mask', self.mask)

        # Register a hook to zero out gradients for masked weights
        self.weight.register_hook(self._zero_grad_for_masked_weights)
        

        # Convert weights and biases to float
        self.weight = nn.Parameter(self.weight.float())
        
        

        # Handle bias initialization if required
        if bias:
            nn.init.uniform_(self.bias, -0.75, 0.75)
            self.bias = nn.Parameter(self.bias.float())  # Ensure biases are float
        
        

    def _zero_grad_for_masked_weights(self, grad):
        """
        Hook to zero out gradients for weights that should not be updated (masked weights).
        """
        return grad * self.update_mask

    def forward(self, input):
        return nn.functional.linear(input, self.weight, self.bias)

    
    
class mask_for_sparsity:
    def __init__(self, n_hid_vars:int, n_obs:int , pct_zeroed_weights:float):
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.pct_zeroed_weights = pct_zeroed_weights
    
    def split_design( self ) -> torch.Tensor:

        """
        Here, we have sectioned the mask tensor (p ,p) into two sub tensors which contains
        the contributions of two parts during transition step
        - A (p,r) : for the observables  : tc_obs
        - A (p,p-r) for the hidden variables : tc_hidvars

        The sparsity will be applied on the tc_hidvars and merged to tc_obs later to 
        form the mask.

        """
        total_weights = self.n_hid_vars * self.n_hid_vars
        zeroed_weights =  int(self.pct_zeroed_weights* total_weights )
        

        tc_obs = torch.ones((self.n_hid_vars,self.n_obs)) #(p,r)
        tc_hidvars = torch.ones((self.n_hid_vars,self.n_hid_vars-self.n_obs))  #(p,p-r)
        total_parms_tc_obs = tc_hidvars.shape[0]*tc_hidvars.shape[1]

        indxes__sparsity = torch.randperm(total_parms_tc_obs)[:zeroed_weights]
        tc_hidvars = tc_hidvars.flatten()

        for idx_ in indxes__sparsity:
            tc_hidvars[idx_.item()] = 0

        tc_hidvars= tc_hidvars.reshape((self.n_hid_vars ,self.n_hid_vars -self.n_obs))
        mask = torch.cat((tc_obs,tc_hidvars), dim=1)
        
        return mask




class LargeSparse_cell(nn.Module):

    

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 
    def __init__(self, n_obs: int, n_hid_vars: int, n_ext_vars : int, pct_zeroed_weights:float):
        super(LargeSparse_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars
        self.pct_zeroed_weights = pct_zeroed_weights
        self.n_ext_vars = n_ext_vars
        sparse_inits =  mask_for_sparsity(n_hid_vars=self.n_hid_vars , n_obs=self.n_obs, pct_zeroed_weights=self.pct_zeroed_weights)
        #self.dtype = torch.float32 
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameter initialization 
        self.B = CustomLinear(in_features =n_ext_vars ,out_features =n_hid_vars)
        self.Sparse_A = CustomLinear(in_features =n_hid_vars ,out_features =n_hid_vars)  
        
        nn.utils.prune.custom_from_mask(self.Sparse_A  , name='weight' , mask = sparse_inits.split_design())

        self.register_buffer('ConMat', torch.eye(n_obs, n_hid_vars, dtype= self.dtype))
        self.register_buffer('Ide', torch.eye(n_hid_vars, dtype=self.dtype))
        self.to(self.device)


    def forward(self, state: torch.Tensor, allow_transition :bool , externals: Optional[torch.Tensor] = None,
                 observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        
        """
        Input Shape:

            - state: (batch_size, 1, n_hid_vars)
            - allow_transition: bool
            - externals: (batch_size, 1, n_ext_vars)
            - drop_output: bool
            - observation: (batch_size, 1, n_obs) or None
        
        Output Shape:
                    
            If observation is not None:

                - If allow_transition is True:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)

                - If allow_transition is False:
                    - expectation: (batch_size, 1, n_obs)
                    - output: (batch_size, 1, n_obs)

            If observation is  None:

                    - expectation: (batch_size, 1, n_obs)
                    - next_state: (batch_size, 1, n_hid_vars)


            - expectation: (batch_size, 1, n_obs)
            - output: (batch_size, 1, n_obs) if drop_output is False, else (batch_size, 1, n_hid_vars)
            - next_state: (batch_size, 1, n_hid_vars) if allow_transition is True, else None
        
        """
        expectation = torch.matmul(state, self.ConMat.T ) 

        if observation is not None:
            
            
                
            output =  expectation - observation
        

            if allow_transition == True:
                
                teach_forc = torch.matmul(output, self.ConMat )#output
                
                r_state = torch.matmul(state , self.Ide) - teach_forc

                next_state = self.Sparse_A( torch.tanh(r_state)) + self.B( externals)

                return expectation, output,  next_state
            
            else: 
                
                return expectation, output


        else:
            
            output = expectation

            r_state = torch.matmul(state , self.Ide)

            next_state = self.Sparse_A( torch.tanh(r_state)) + self.B( externals)

        
        return expectation, output,  next_state 

