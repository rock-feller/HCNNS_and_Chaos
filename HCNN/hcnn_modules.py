import torch
from typing import Tuple


import torch
from typing import Optional , Tuple
import torch.nn as nn

class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 init_range: Tuple[float, float] = (-0.75, 0.75)):
        """CustomLinear layer with user-defined weight initialization range."""
        self.init_range = init_range
        super(CustomLinear, self).__init__(in_features, out_features, bias=bias)

        # Properly initialize weights and biases
        nn.init.uniform_(self.weight.data, self.init_range[0], self.init_range[1])
        if bias:
            nn.init.uniform_(self.bias.data, self.init_range[0], self.init_range[1])


class vanilla_hcnncell(nn.Module):
    """
    Vanilla HCNN Cell.

    This class implements the Vanilla HCNN Cell for state-based predictions and state transitions.
    It supports the use of teacher forcing during training and provides methods to compute the 
    next state and predicted outputs.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    A : CustomLinear
        Linear transformation module for updating the hidden state.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]
        Performs a forward pass through the Vanilla HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """


    def __init__(self, n_obs: int, 
                 n_hid_vars: int , 
                 init_range: Tuple[float,float] = (-0.75,0.75)):

        super(vanilla_hcnncell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Parameter initialization 
         
        self.A = CustomLinear(in_features =n_hid_vars, out_features =n_hid_vars , bias = False ,init_range = init_range)

        self.register_buffer(name = 'ConMat', tensor= torch.eye(n_obs, n_hid_vars), persistent = False)
        
        self.register_buffer(name = 'Ide', tensor= torch.eye(n_hid_vars ), persistent = False)

        # Select device
        self.device = self._get_default_device()
        
    def _get_default_device(self) -> torch.device:

        """
        Determines the default device to use for computations.

        Returns
        -------
        torch.device
            The default device (`cuda`, `mps`, or `cpu`).
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


    def forward(self, state: torch.Tensor, teacher_forcing: bool,
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Vanilla HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor | shape=(n_hid_vars,)
            The current state tensor  `s_{t}`is the HCNN state vector of shape (n_hid_vars,)
        teacher_forcing : bool
            Whether to use teacher forcing for the state transition:
            - True: Use the provided observation (`observation`) to guide the state transition.
            - False: Compute the next state based on the model's prediction.
        observation : Optional[torch.Tensor], default=None | shape=(n_obs,)
            The ground-truth observation tensor (`y_true`) corresponds to the observable at time t. Required when 
            `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - expectation : torch.Tensor
                Predicted observation tensor (`y_hat`) |  shape (n_obs,).
            - next_state : torch.Tensor
                Updated state tensor (`s_{t+1}`) |  shape (n_hid_vars,).
            - delta_term : `y_hat - y_true` | shape (n_obs,) | Optional , can be None.

        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the method uses `observation` to compute a correction term 
          (`delta_term`) for guiding the state transition.
          
        - If `teacher_forcing` is False, the state transition is based purely on the model's 
          internal computation.
        """
    # Compute the expected output (y_hat)
        expectation = torch.matmul(self.ConMat, state)

        print("expectation requires_grad:", expectation.requires_grad)
        if teacher_forcing:

            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")

            # Compute the delta term (y_true - y_hat)
            delta_term = observation - expectation

            # Teacher forcing: Correct the state using the delta term
            teach_forc = torch.matmul(self.ConMat.T, delta_term)

            r_state = state - teach_forc
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, delta_term
        else:
            # Without teacher forcing: State evolves independently
            r_state = torch.matmul(self.Ide, state)
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, None
