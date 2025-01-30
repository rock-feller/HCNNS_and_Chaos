import torch
import pytest
from typing import Tuple



import torch
from typing import Tuple
from torch.nn import MSELoss


import torch
from typing import Optional , Tuple
import torch.nn as nn


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


class vanilla_cell(nn.Module):
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

        super(vanilla_cell, self).__init__()
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
        # expectation = torch.matmul(self.ConMat, state)

        expectation = torch.matmul(state , self.ConMat.T)

        print("expectation requires_grad:", expectation.requires_grad)
        if teacher_forcing:

            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")

            # Compute the delta term (y_true - y_hat)
            delta_term = observation - expectation

            # Teacher forcing: Correct the state using the delta term
            teach_forc = torch.matmul(delta_term,self.ConMat)

            r_state = state - teach_forc
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, delta_term
        else:
            # Without teacher forcing: State evolves independently
            r_state = torch.matmul(self.Ide, state)
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, None
        
        # Define a simple test suite
def test_initialization():
    """Test if the VanillaHCNNCell initializes correctly."""
    n_obs = 5
    n_hid_vars = 10
    cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Check attribute correctness
    assert cell.n_obs == n_obs, "Number of observed variables is incorrect."
    assert cell.n_hid_vars == n_hid_vars, "Number of hidden variables is incorrect."
    assert cell.ConMat.shape == (n_obs, n_hid_vars), "Connection matrix shape is incorrect."
    assert cell.Ide.shape == (n_hid_vars, n_hid_vars), "Identity matrix shape is incorrect."
    assert isinstance(cell.A, CustomLinear), "CustomLinear layer is not correctly initialized."

def test_device_assignment():
    """Test if the default device is assigned correctly."""
    cell = vanilla_cell(5, 10)
    expected_device = cell._get_default_device()
    assert cell.device == expected_device, "Device assignment is incorrect."

def test_forward_no_teacher_forcing():
    """Test the forward method without teacher forcing."""
    n_obs, n_hid_vars = 5, 10
    cell = vanilla_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    expectation, next_state, delta_term = cell(state, teacher_forcing=False)

    # Check shapes of outputs
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert delta_term is None, "Delta term should be None when teacher forcing is False."

def test_forward_with_teacher_forcing():
    """Test the forward method with teacher forcing."""
    n_obs, n_hid_vars = 5, 10
    cell = vanilla_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation
    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    # Check shapes of outputs
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert delta_term.shape == (n_obs,), "Delta term shape is incorrect."

def test_teacher_forcing_without_observation():
    """Test if the method raises an error when teacher forcing is enabled but observation is missing."""
    n_obs, n_hid_vars = 5, 10
    cell = vanilla_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state

    with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
        _ = cell(state, teacher_forcing=True)

def test_consistency_in_predictions():
    """Test if the forward pass produces consistent predictions given the same inputs."""
    n_obs, n_hid_vars = 5, 10
    cell = vanilla_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation

    output_1 = cell(state, teacher_forcing=True, observation=observation)
    output_2 = cell(state, teacher_forcing=True, observation=observation)

    for o1, o2 in zip(output_1, output_2):
        if o1 is not None:
            assert torch.allclose(o1, o2), "Outputs are inconsistent for identical inputs."

def test_gradient_flow():
    """Test if gradients flow properly through the model."""
    
    n_obs, n_hid_vars = 5, 10
    cell = vanilla_cell(n_obs, n_hid_vars)
    print("Weight requires_grad:", cell.A.weight.requires_grad)
    print("Weight gradient before backward:", cell.A.weight.grad)

    state = torch.randn(n_hid_vars, requires_grad=True)  # Enable gradient tracking
    observation = torch.randn(n_obs)  # Random observation
    some_target_state = torch.randn(n_hid_vars)
    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    #loss = torch.sum((expectation - observation) ** 2) + torch.sum(next_state ** 2) # Mean Squared Error-like loss
    loss_fct =  torch.nn.MSELoss()
    # loss = loss_fct(expectation, observation)
    loss_fct = loss_fct(some_target_state, next_state)
    loss_fct.backward()  # Backpropagation

    assert state.grad is not None, "Gradients are not flowing through the state."
    assert cell.A.weight.grad is not None, "Gradients are not flowing through the CustomLinear weights."

def test_scalability():
    """Test if the module works with larger inputs."""
    n_obs, n_hid_vars = 100, 200  # Larger sizes
    cell = vanilla_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation

    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    # Check shapes of outputs
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect for large inputs."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect for large inputs."
    assert delta_term.shape == (n_obs,), "Delta term shape is incorrect for large inputs."


def test_gradient_flow_back_to_So_A():
    """Test vanilla_hcnncell in a recurrent fashion over a sequence."""
    T, n_obs, n_hid_vars = 5, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Initialize input sequence (T, n_obs)
    tens_ytrues =  torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
    loss_fct = torch.nn.MSELoss() 
    tens_yhats =  torch.empty(T, n_obs)
    tens_delta_terms =  torch.empty(T, n_obs)
    tens_states =  torch.empty(T, n_hid_vars)

    tens_states[0] = state

    for time_step in range(T -1):
        tens_yhats[time_step], tens_states[time_step], delta_term = cell.forward(state= tens_states[time_step],
                                                                teacher_forcing=True,
                                                                observation=tens_ytrues[time_step])

    tens_yhats[time_step] =torch.matmul(cell.ConMat, tens_states[time_step])                                                       
    
    tens_delta_terms[time_step] = tens_ytrues[time_step]  - tens_yhats[time_step]

                                
    loss =  loss_fct(tens_yhats, tens_ytrues) 
    loss.backward()  # Backpropagation

    print("State gradient:", state.grad)
    print("A weight gradient:", cell.A.weight.grad)

    assert state.grad is not None, "Gradients are not flowing through the initial state."
    assert cell.A.weight.grad is not None, "Gradients are not flowing through the CustomLinear weights."




def test_vanilla_hcnncell_recurrent():
    """
    Test vanilla_hcnncell in a recurrent fashion over a sequence and verify correctness
    Here, we uselists to collect intermediate results ensures that each tensor 
    retains its link to the computational graph.
    It can be visualized using `torchviz.make_dot`

    make_dot(loss, params = {"A": cell.A.weight,
                            "ConMat": cell.ConMat, "Ide": cell.Ide,
                            "loss term": loss, "state_s0": state}).
    
    """
    # Define test parameters
    T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Initialize input sequence (T, n_obs)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
    loss_fct = MSELoss()

    # Use lists instead of pre-allocated tensors
    list_yhats = []        # To store y_hat for each time step
    list_delta_terms = []  # To store delta terms for each time step
    list_states = [state]  # To store states (initialize with the first state)

    # Forward pass over the sequence
    for time_step in range(T - 1):
        y_hat, next_state, delta_term = cell.forward(
            state=list_states[time_step],
            teacher_forcing=True,
            observation=tens_ytrues[time_step]
        )
        list_yhats.append(y_hat)
        list_delta_terms.append(delta_term)
        list_states.append(next_state)

    # Compute y_hat for the last time step
    last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
    last_delta_term = tens_ytrues[-1] - last_y_hat

    list_yhats.append(last_y_hat)
    list_delta_terms.append(last_delta_term)

    # Stack lists into tensors
    tens_yhats = torch.stack(list_yhats)           # Shape: (T, n_obs)
    tens_delta_terms = torch.stack(list_delta_terms)  # Shape: (T, n_obs)
    tens_states = torch.stack(list_states)         # Shape: (T, n_hid_vars)

    # Compute loss
    loss = loss_fct(tens_yhats, tens_ytrues)

    # Backward pass
    loss.backward()

    # Assertions to verify correctness
    assert tens_yhats.shape == (T, n_obs), "Output tensor `tens_yhats` has an incorrect shape."
    assert tens_delta_terms.shape == (T, n_obs), "Delta term tensor `tens_delta_terms` has an incorrect shape."
    assert tens_states.shape == (T, n_hid_vars), "State tensor `tens_states` has an incorrect shape."
    assert state.grad is not None, "Gradients are not flowing through the initial state."
    assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
    assert loss.item() > 0, "Loss should be a positive scalar."
    
    # Additional checks
    print("Test passed: all checks are valid!")



import torch.nn as nn

def log_cosh_loss(y_hat, y_true):
    """
    Computes the Log-Cosh Loss for the given predictions and targets.

    Args:
        y_hat (torch.Tensor): Predicted values (shape: [T, n_obs]).
        y_true (torch.Tensor): Ground-truth values (shape: [T, n_obs]).

    Returns:
        torch.Tensor: Scalar Log-Cosh loss.
    """
    diff = y_hat - y_true
    return torch.mean(torch.log(torch.cosh(diff)))

def test_vanilla_hcnncell_logcosh():
    """
    Test vanilla_hcnncell using the Log-Cosh loss function and ensure gradient flow.
    """
    # Define test parameters
    T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Initialize input sequence (T, n_obs)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state

    # Use lists instead of pre-allocated tensors
    list_yhats = []        # To store y_hat for each time step
    list_states = [state]  # To store states (initialize with the first state)

    # Forward pass over the sequence
    for time_step in range(T - 1):
        y_hat, next_state, _ = cell.forward(
            state=list_states[time_step],
            teacher_forcing=True,
            observation=tens_ytrues[time_step]
        )
        list_yhats.append(y_hat)
        list_states.append(next_state)

    # Compute y_hat for the last time step
    last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
    list_yhats.append(last_y_hat)

    # Stack lists into tensors
    tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
    tens_states = torch.stack(list_states)  # Shape: (T, n_hid_vars)

    # Compute Log-Cosh loss
    loss = log_cosh_loss(tens_yhats, tens_ytrues)

    # Backward pass
    loss.backward()

    # Assertions to verify correctness
    assert tens_yhats.shape == (T, n_obs), "Output tensor `tens_yhats` has an incorrect shape."
    assert tens_states.shape == (T, n_hid_vars), "State tensor `tens_states` has an incorrect shape."
    assert state.grad is not None, "Gradients are not flowing through the initial state."
    assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
    assert loss.item() > 0, "Loss should be a positive scalar."

    # Additional checks
    print("Log-Cosh Loss:", loss.item())
    print("Test passed: all checks are valid!")


# # Define a simple test suite
# def test_initialization():
#     """Test if the VanillaHCNNCell initializes correctly."""
#     n_obs = 5
#     n_hid_vars = 10
#     cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

#     # Check attribute correctness
#     assert cell.n_obs == n_obs, "Number of observed variables is incorrect."
#     assert cell.n_hid_vars == n_hid_vars, "Number of hidden variables is incorrect."
#     assert cell.ConMat.shape == (n_obs, n_hid_vars), "Connection matrix shape is incorrect."
#     assert cell.Ide.shape == (n_hid_vars, n_hid_vars), "Identity matrix shape is incorrect."
#     assert isinstance(cell.A, CustomLinear), "CustomLinear layer is not correctly initialized."

# def test_device_assignment():
#     """Test if the default device is assigned correctly."""
#     cell = vanilla_cell(5, 10)
#     expected_device = cell._get_default_device()
#     assert cell.device == expected_device, "Device assignment is incorrect."

# def test_forward_no_teacher_forcing():
#     """Test the forward method without teacher forcing."""
#     n_obs, n_hid_vars = 5, 10
#     cell = vanilla_cell(n_obs, n_hid_vars)

#     state = torch.randn(n_hid_vars)  # Random initial state
#     expectation, next_state, delta_term = cell(state, teacher_forcing=False)

#     # Check shapes of outputs
#     assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
#     assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
#     assert delta_term is None, "Delta term should be None when teacher forcing is False."

# def test_forward_with_teacher_forcing():
#     """Test the forward method with teacher forcing."""
#     n_obs, n_hid_vars = 5, 10
#     cell = vanilla_cell(n_obs, n_hid_vars)

#     state = torch.randn(n_hid_vars)  # Random initial state
#     observation = torch.randn(n_obs)  # Random observation
#     expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

#     # Check shapes of outputs
#     assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
#     assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
#     assert delta_term.shape == (n_obs,), "Delta term shape is incorrect."

# def test_teacher_forcing_without_observation():
#     """Test if the method raises an error when teacher forcing is enabled but observation is missing."""
#     n_obs, n_hid_vars = 5, 10
#     cell = vanilla_cell(n_obs, n_hid_vars)

#     state = torch.randn(n_hid_vars)  # Random initial state

#     with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
#         _ = cell(state, teacher_forcing=True)

# def test_consistency_in_predictions():
#     """Test if the forward pass produces consistent predictions given the same inputs."""
#     n_obs, n_hid_vars = 5, 10
#     cell = vanilla_cell(n_obs, n_hid_vars)

#     state = torch.randn(n_hid_vars)  # Random initial state
#     observation = torch.randn(n_obs)  # Random observation

#     output_1 = cell(state, teacher_forcing=True, observation=observation)
#     output_2 = cell(state, teacher_forcing=True, observation=observation)

#     for o1, o2 in zip(output_1, output_2):
#         if o1 is not None:
#             assert torch.allclose(o1, o2), "Outputs are inconsistent for identical inputs."

# def test_gradient_flow():
#     """Test if gradients flow properly through the model."""
    
#     n_obs, n_hid_vars = 5, 10
#     cell = vanilla_cell(n_obs, n_hid_vars)
#     print("Weight requires_grad:", cell.A.weight.requires_grad)
#     print("Weight gradient before backward:", cell.A.weight.grad)

#     state = torch.randn(n_hid_vars, requires_grad=True)  # Enable gradient tracking
#     observation = torch.randn(n_obs)  # Random observation

#     expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

#     #loss = torch.sum((expectation - observation) ** 2) + torch.sum(next_state ** 2) # Mean Squared Error-like loss
#     loss_fct =  torch.nn.MSELoss()
#     loss = loss_fct(expectation, observation)
#     loss.backward()  # Backpropagation

#     assert state.grad is not None, "Gradients are not flowing through the state."
#     assert cell.A.weight.grad is None, "Expected as Gradients are not flowing through the CustomLinear weights."

# def test_scalability():
#     """Test if the module works with larger inputs."""
#     n_obs, n_hid_vars = 100, 200  # Larger sizes
#     cell = vanilla_cell(n_obs, n_hid_vars)

#     state = torch.randn(n_hid_vars)  # Random initial state
#     observation = torch.randn(n_obs)  # Random observation

#     expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

#     # Check shapes of outputs
#     assert expectation.shape == (n_obs,), "Output expectation shape is incorrect for large inputs."
#     assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect for large inputs."
#     assert delta_term.shape == (n_obs,), "Delta term shape is incorrect for large inputs."


# def test_gradient_flow_back_to_So_A():
#     """Test vanilla_hcnncell in a recurrent fashion over a sequence."""
#     T, n_obs, n_hid_vars = 5, 4, 6  # Number of timesteps, observed vars, hidden vars
#     cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

#     # Initialize input sequence (T, n_obs)
#     tens_ytrues =  torch.randn(T, n_obs)  # Random sequence of observations
#     state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
#     loss_fct = torch.nn.MSELoss() 
#     tens_yhats =  torch.empty(T, n_obs)
#     tens_delta_terms =  torch.empty(T, n_obs)
#     tens_states =  torch.empty(T, n_hid_vars)

#     tens_states[0] = state

#     for time_step in range(T -1):
#         tens_yhats[time_step], tens_states[time_step], delta_term = cell.forward(state= tens_states[time_step],
#                                                               teacher_forcing=True,
#                                                                 observation=tens_ytrues[time_step])

#     tens_yhats[time_step] =torch.matmul(cell.ConMat, tens_states[time_step])                                                       
    
#     tens_delta_terms[time_step] = tens_ytrues[time_step]  - tens_yhats[time_step]

                              
#     loss =  loss_fct(tens_yhats, tens_ytrues) 
#     loss.backward()  # Backpropagation

#     print("State gradient:", state.grad)
#     print("A weight gradient:", cell.A.weight.grad)

#     assert state.grad is not None, "Gradients are not flowing through the initial state."
#     assert cell.A.weight.grad is not None, "Gradients are not flowing through the CustomLinear weights."




# def test_vanilla_hcnncell_recurrent():
#     """
#     Test vanilla_hcnncell in a recurrent fashion over a sequence and verify correctness
#     Here, we uselists to collect intermediate results ensures that each tensor 
#     retains its link to the computational graph.
#     It can be visualized using `torchviz.make_dot`

#     make_dot(loss, params = {"A": cell.A.weight,
#                             "ConMat": cell.ConMat, "Ide": cell.Ide,
#                             "loss term": loss, "state_s0": state}).
    
#     """
#     # Define test parameters
#     T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
#     cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

#     # Initialize input sequence (T, n_obs)
#     tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
#     state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
#     loss_fct = MSELoss()

#     # Use lists instead of pre-allocated tensors
#     list_yhats = []        # To store y_hat for each time step
#     list_delta_terms = []  # To store delta terms for each time step
#     list_states = [state]  # To store states (initialize with the first state)

#     # Forward pass over the sequence
#     for time_step in range(T - 1):
#         y_hat, next_state, delta_term = cell.forward(
#             state=list_states[time_step],
#             teacher_forcing=True,
#             observation=tens_ytrues[time_step]
#         )
#         list_yhats.append(y_hat)
#         list_delta_terms.append(delta_term)
#         list_states.append(next_state)

#     # Compute y_hat for the last time step
#     last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
#     last_delta_term = tens_ytrues[-1] - last_y_hat

#     list_yhats.append(last_y_hat)
#     list_delta_terms.append(last_delta_term)

#     # Stack lists into tensors
#     tens_yhats = torch.stack(list_yhats)           # Shape: (T, n_obs)
#     tens_delta_terms = torch.stack(list_delta_terms)  # Shape: (T, n_obs)
#     tens_states = torch.stack(list_states)         # Shape: (T, n_hid_vars)

#     # Compute loss
#     loss = loss_fct(tens_yhats, tens_ytrues)

#     # Backward pass
#     loss.backward()

#     # Assertions to verify correctness
#     assert tens_yhats.shape == (T, n_obs), "Output tensor `tens_yhats` has an incorrect shape."
#     assert tens_delta_terms.shape == (T, n_obs), "Delta term tensor `tens_delta_terms` has an incorrect shape."
#     assert tens_states.shape == (T, n_hid_vars), "State tensor `tens_states` has an incorrect shape."
#     assert state.grad is not None, "Gradients are not flowing through the initial state."
#     assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
#     assert loss.item() > 0, "Loss should be a positive scalar."
    
#     # Additional checks
#     print("Test passed: all checks are valid!")



# import torch
# import torch.nn as nn

# def log_cosh_loss(y_hat, y_true):
#     """
#     Computes the Log-Cosh Loss for the given predictions and targets.

#     Args:
#         y_hat (torch.Tensor): Predicted values (shape: [T, n_obs]).
#         y_true (torch.Tensor): Ground-truth values (shape: [T, n_obs]).

#     Returns:
#         torch.Tensor: Scalar Log-Cosh loss.
#     """
#     diff = y_hat - y_true
#     return torch.mean(torch.log(torch.cosh(diff)))

# def test_vanilla_hcnncell_logcosh():
#     """
#     Test vanilla_hcnncell using the Log-Cosh loss function and ensure gradient flow.
#     """
#     # Define test parameters
#     T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
#     cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

#     # Initialize input sequence (T, n_obs)
#     tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
#     state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state

#     # Use lists instead of pre-allocated tensors
#     list_yhats = []        # To store y_hat for each time step
#     list_states = [state]  # To store states (initialize with the first state)

#     # Forward pass over the sequence
#     for time_step in range(T - 1):
#         y_hat, next_state, _ = cell.forward(
#             state=list_states[time_step],
#             teacher_forcing=True,
#             observation=tens_ytrues[time_step]
#         )
#         list_yhats.append(y_hat)
#         list_states.append(next_state)

#     # Compute y_hat for the last time step
#     last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
#     list_yhats.append(last_y_hat)

#     # Stack lists into tensors
#     tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
#     tens_states = torch.stack(list_states)  # Shape: (T, n_hid_vars)

#     # Compute Log-Cosh loss
#     loss = log_cosh_loss(tens_yhats, tens_ytrues)

#     # Backward pass
#     loss.backward()

#     # Assertions to verify correctness
#     assert tens_yhats.shape == (T, n_obs), "Output tensor `tens_yhats` has an incorrect shape."
#     assert tens_states.shape == (T, n_hid_vars), "State tensor `tens_states` has an incorrect shape."
#     assert state.grad is not None, "Gradients are not flowing through the initial state."
#     assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
#     assert loss.item() > 0, "Loss should be a positive scalar."

#     # Additional checks
#     print("Log-Cosh Loss:", loss.item())
#     print("Test passed: all checks are valid!")


