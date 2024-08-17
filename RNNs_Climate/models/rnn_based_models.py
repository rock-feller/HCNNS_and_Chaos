import torch
import torch.nn as nn
from typing import Tuple, Literal

class RNNModelClimate(nn.Module):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 s0_nature: Literal['zeros_', 'random_'], train_s0: bool, num_layers=1):
        super(RNNModelClimate, self).__init__()
        self.num_layers = num_layers
        self.train_s0 = train_s0
        self.hidden_size = hidden_size
        self.s0_nature = s0_nature
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.state_transition_layer = nn.RNN(input_size, hidden_size, batch_first=True,
                                             num_layers=num_layers, bias=False , device=self.device)

        self.output_extraction_layer = nn.Linear(hidden_size, output_size, bias=False, device=self.device)

    def initial_hidden_state(self, batch_size: int) -> torch.Tensor:
        if self.s0_nature.lower() == "zeros_":
            return nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)
        elif self.s0_nature.lower() == "random_":
            torch.manual_seed(0)  # Set seed for reproducibility
            return nn.Parameter(torch.empty(self.num_layers, batch_size, self.hidden_size, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")

    def forward(self, input_sequence: torch.Tensor, initial_states_s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """Inputs shape:
          input_sequence: [batch, seq_len, features]
          initial_states_s0: [num_layers, batch, hidden_size]



        The state_transition equation computes the hidden states of the RNN
        across the whole batch of sequences and stack them the following way
        [S_(1), ...., S_(t-2), S_(t-1), S_(t)] where S_(t) is the hidden state at time t
        Then, the outpute equation extracts the output from the hidden states
        and return the outputs as  a batch of [y_(0), ...., y_(t-2), y_(t-1), y_(t)]

        Outputs shape:
        sequence_of_states: [batch, seq_len, hidden_size]
        output_sequence: [batch, seq_len, features]
        hstate_at_T: [num_layers, batch, hidden_size]
        """
                
        sequence_of_states, hstate_at_T = self.state_transition_layer(input_sequence, initial_states_s0)
        output_sequence = self.output_extraction_layer(sequence_of_states)
        return sequence_of_states, output_sequence, hstate_at_T





class LSTMModelClimate(nn.Module):

    
    dtype = torch.float32

    def __init__(self, input_size: int, hidden_size: int, output_size: int, s0_nature: Literal['zeros_', 'random_'],
                 train_s0: bool, num_layers=1):
        super(LSTMModelClimate, self).__init__()

        self.num_layers = num_layers
        self.train_s0 = train_s0
        self.s0_nature = s0_nature
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        
        self.state_transition_layer = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, bias=True , device=self.device)
        self.output_extraction_layer = nn.Linear(hidden_size, output_size, dtype=self.dtype, bias=True, device=self.device)

    def initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.s0_nature.lower() == "zeros_":
            state_h0 = nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)
            state_c0 = nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=self.dtype, device=self.device), requires_grad=self.train_s0)
        elif self.s0_nature.lower() == "random_":
            torch.manual_seed(0)  # Set seed for reproducibility
            state_h0 = nn.Parameter(torch.empty(self.num_layers, batch_size, self.hidden_size, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
            state_c0 = nn.Parameter(torch.empty(self.num_layers, batch_size, self.hidden_size, dtype=self.dtype, device=self.device).uniform_(-0.75, 0.75), requires_grad=self.train_s0)
        else:
            raise ValueError("s0_nature must be either 'zeros_' or 'random_'")
        
        return state_h0, state_c0

    def forward(self, input_sequence: torch.Tensor, h0_c0_tuple: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        """
        Assuming the input_sequence is of shape [batch, window, features]
        The state_transition equation computes the hidden states of the LSTM
        across the whole batch of sequences and stack them the following way
        [S_(1), ...., S_(t-2), S_(t-1), S_(t)] where S_(t) is the hidden state at time t
        Then, the outpute equation extracts the output from the hidden states
        and return the outputs as  a batch of [y_(0), ...., y_(t-2), y_(t-1), y_(t)]
        """
        
        sequence_of_states, (hstate_at_T, cstate_at_T) = self.state_transition_layer(input_sequence, h0_c0_tuple)
        output_sequence = self.output_extraction_layer(sequence_of_states)
        return sequence_of_states, output_sequence, (hstate_at_T, cstate_at_T)
        
        
        
       
