import torch
import pytest
from torch.nn import MSELoss
import torch.nn as nn
from typing import Optional , Tuple, List

class DiagonalMatrix(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, init_diag: Optional[float] = 1.0):
        if in_features != out_features:
            raise ValueError("DiagonalMatrix requires in_features to be equal to out_features for symmetry.")
        super(DiagonalMatrix, self).__init__(in_features, out_features, bias=bias)

        # Initialize the weight matrix
        nn.init.constant_(self.weight, 0)  # Set all weights to zero
        if init_diag is not None:
            self.weight.data.fill_diagonal_(init_diag)  # User-specified or default value
        else:
            nn.init.uniform_(self.weight.data.fill_diagonal_(0), -1e-5, 1e-5)  # Random diagonal values


        # Pre-compute and store the diagonal mask
        self.register_buffer("mask", torch.eye(self.weight.shape[0], device=self.weight.device))

        # Register hook
        self.weight.register_hook(self._clamp_and_zero_out)


    def _clamp_and_zero_out(self, grad):

        """
        Zero out off-diagonal elements and clamp diagonal values
        Grad here, gradient_loss_wrt_weight.
        """
        with torch.no_grad():
            self.weight.data = torch.mul(self.weight.data ,self.mask)  # Retain only diagonal elements
            self.weight.data.clamp_(min=0., max=1)

        return grad * self.mask  # Mask gradients to zero out off-diagonal elements
    

    def forward(self, input, verbose: bool = False):
            if verbose:
                print(f"Diagonal weights: {torch.diag(self.weight)}")
            return nn.functional.linear(input, self.weight, self.bias)



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



class lstm_cell(nn.Module):
    """
    LSTM Formulation of the HCNN (LForm HCNN Cell).

    This class implements the LSTM-inspired formulation of the HCNN Cell for state-based predictions and state transitions.
    It incorporates a custom diagonal matrix transformation to enforce exponential embedding of the residual state.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    A : CustomLinear
        Linear transformation module for processing the non-linear residual state.
    D : DiagonalMatrix
        Diagonal matrix transformation module for controlling the residual state dynamics.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    _get_default_device() -> torch.device:
        Determines the default device for computations.
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Performs a forward pass through the LSTM HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """



    def __init__(self, n_obs: int, 
                 n_hid_vars: int , 
                 init_range: Tuple[float,float] = (-0.75,0.75)):

        super(lstm_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Parameter initialization 
         
        self.A = CustomLinear(in_features = n_hid_vars, out_features =n_hid_vars , bias = False ,init_range = init_range)
        
        self.D = DiagonalMatrix(in_features =n_hid_vars, out_features =n_hid_vars , bias = False ,init_diag = 1.)

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
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the LSTM HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor, shape=(n_hid_vars,)
            The current state tensor (`s_t`) of the HCNN, with shape `(n_hid_vars,)`.
        teacher_forcing : bool
            Whether to use teacher forcing for the state transition:
            - True: Use the provided observation (`observation`) to guide the state transition.
            - False: Compute the next state based on the model's prediction.
        observation : Optional[torch.Tensor], default=None, shape=(n_obs,)
            The ground-truth observation tensor (`y_true`) for time step `t`.
            Required when `teacher_forcing` is True. Ignored otherwise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - expectation : torch.Tensor, shape=(n_obs,)
                Predicted observation tensor (`y_hat`).
            - next_state : torch.Tensor, shape=(n_hid_vars,)
                Updated state tensor (`s_{t+1}`).
            - delta_term : Optional[torch.Tensor], shape=(n_obs,)
                Difference between the prediction and ground-truth (`y_hat - y_true`).
                Returns None when `teacher_forcing` is False.

        Raises
        ------
        ValueError
            If `teacher_forcing` is True and `observation` is not provided.

        Notes
        -----
        - When `teacher_forcing` is True, the method uses `observation` to compute a correction term 
        (`delta_term`) for guiding the state transition.
        - If `teacher_forcing` is False, the state transition is based purely on the model's prediction.

        - State Transition Logic:
            - Compute the residual state (`r_state`).
            - Apply a non-linear transformation using the `CustomLinear` layer (`A`).
            - Add an adjusted residual term using the diagonal matrix (`D`).
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

            lstm_block = self.A(torch.tanh(r_state)) -  r_state

            next_state = r_state + self.D(lstm_block)


            return expectation, next_state, delta_term
        else:
            # Without teacher forcing: State evolves independently
            r_state = torch.matmul(self.Ide, state)
            
            lstm_block = self.A(torch.tanh(r_state)) -  r_state

            next_state = r_state + self.D(lstm_block)


            return expectation, next_state, None


class DiagonalMatrix(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, init_diag: Optional[float] = 1.0):
        if in_features != out_features:
            raise ValueError("DiagonalMatrix requires in_features to be equal to out_features for symmetry.")
        super(DiagonalMatrix, self).__init__(in_features, out_features, bias=bias)

        # Initialize the weight matrix
        nn.init.constant_(self.weight, 0)  # Set all weights to zero
        if init_diag is not None:
            self.weight.data.fill_diagonal_(init_diag)  # User-specified or default value
        else:
            nn.init.uniform_(self.weight.data.fill_diagonal_(0), -1e-5, 1e-5)  # Random diagonal values


        # Pre-compute and store the diagonal mask
        self.register_buffer("mask", torch.eye(self.weight.shape[0], device=self.weight.device))

        # Register hook
        self.weight.register_hook(self._clamp_and_zero_out)


    def _clamp_and_zero_out(self, grad):

        """
        Zero out off-diagonal elements and clamp diagonal values
        Grad here, gradient_loss_wrt_weight.
        """
        with torch.no_grad():
            self.weight.data = torch.mul(self.weight.data ,self.mask)  # Retain only diagonal elements
            self.weight.data.clamp_(min=0., max=1)

        return grad * self.mask  # Mask gradients to zero out off-diagonal elements
    

    def forward(self, input, verbose: bool = False):
            if verbose:
                print(f"Diagonal weights: {torch.diag(self.weight)}")
            return nn.functional.linear(input, self.weight, self.bias)



def test_initialization():
    """Test if the LSTM HCNN Cell initializes correctly."""
    n_obs = 5
    n_hid_vars = 10
    cell = lstm_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Check attribute correctness
    assert cell.n_obs == n_obs, "Number of observed variables is incorrect."
    assert cell.n_hid_vars == n_hid_vars, "Number of hidden variables is incorrect."
    assert cell.ConMat.shape == (n_obs, n_hid_vars), "Connection matrix shape is incorrect."
    assert cell.Ide.shape == (n_hid_vars, n_hid_vars), "Identity matrix shape is incorrect."
    assert isinstance(cell.A, CustomLinear), "CustomLinear layer is not correctly initialized."
    assert isinstance(cell.D, DiagonalMatrix), "DiagonalMatrix is not correctly initialized."


def test_device_assignment():
    """Test if the default device is assigned correctly."""
    cell = lstm_cell(5, 10)
    expected_device = cell._get_default_device()
    assert cell.device == expected_device, "Device assignment is incorrect."


def test_forward_no_teacher_forcing():
    """Test the forward method without teacher forcing."""
    n_obs, n_hid_vars = 5, 10
    cell = lstm_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    expectation, next_state, delta_term = cell(state, teacher_forcing=False)

    # Check shapes of outputs
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert delta_term is None, "Delta term should be None when teacher forcing is False."


def test_forward_with_teacher_forcing():
    """Test the forward method with teacher forcing."""
    n_obs, n_hid_vars = 5, 10
    cell = lstm_cell(n_obs, n_hid_vars)

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
    cell = lstm_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state

    with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
        _ = cell(state, teacher_forcing=True)


def test_trainable_diagonal_matrix():
    """Test if the diagonal matrix D is trainable and updated during backpropagation."""
    n_obs, n_hid_vars = 5, 10
    cell = lstm_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial state
    some_target_state =  torch.randn(n_hid_vars)  # Random target state
    observation = torch.randn(n_obs)  # Observation
    optimizer = torch.optim.SGD(cell.parameters(), lr=0.01)

    for epoch in range(3):
        optimizer.zero_grad()

        # Forward pass
        expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

        
        # Compute loss
        loss = MSELoss()(next_state, some_target_state)
        
        loss.backward()  # Backpropagation

        # Check if the diagonal matrix gradients are computed
        assert cell.D.weight.grad is not None, "Gradients are not flowing through the DiagonalMatrix."
        assert torch.allclose(cell.D.weight.grad, cell.D.weight.grad * cell.D.mask), \
            "Off-diagonal gradients in DiagonalMatrix are not clamped."

        optimizer.step()

        # Check if the weights of the diagonal matrix are updated
        assert torch.any(cell.D.weight.data != torch.eye(n_hid_vars)), \
            "Diagonal matrix weights are not being updated during training."


def test_gradient_flow():
    """Test if gradients flow properly through all trainable parameters."""

    n_obs, n_hid_vars = 5, 10
    cell = lstm_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars, requires_grad=True)  # Enable gradient tracking
    some_target_state =  torch.randn(n_hid_vars)  # Random target state
    observation = torch.randn(n_obs)  # Random observation

    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    # Compute loss
    loss = MSELoss()(next_state, some_target_state)
    loss.backward()  # Backpropagation

    assert state.grad is not None, "Gradients are not flowing through the initial state."
    assert cell.A.weight.grad is not None, "Gradients are not flowing through the CustomLinear weights."
    assert cell.D.weight.grad is not None, "Gradients are not flowing through the DiagonalMatrix weights."


def test_recurrent_behavior():
    """Test the LSTM HCNN Cell in a recurrent setup over a sequence."""
    T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = lstm_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Initialize input sequence (T, n_obs)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
    loss_fct = MSELoss()

    # Use lists to store outputs
    list_yhats = []
    list_states = [state]

    for time_step in range(T - 1):
        y_hat, next_state, _ = cell(
            state=list_states[time_step],
            teacher_forcing=True,
            observation=tens_ytrues[time_step]
        )
        list_yhats.append(y_hat)
        list_states.append(next_state)

    # Compute y_hat for the last time step
    last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
    list_yhats.append(last_y_hat)

    # Stack and compute loss
    tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
    loss = loss_fct(tens_yhats, tens_ytrues)
    loss.backward()

    # Assertions
    assert state.grad is not None, "Gradients are not flowing through the initial state."
    assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
    assert cell.D.weight.grad is not None, "Gradients are not flowing through the `DiagonalMatrix` weights."
    assert loss.item() > 0, "Loss should be a positive scalar."


def test_recurrent_behavior_across_epochs():
    """Test the LSTM HCNN Cell in a recurrent setup over a sequence."""
    T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = lstm_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Initialize input sequence (T, n_obs)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
    loss_fct = MSELoss()
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.01)

    # Use lists to store outputs
    list_yhats = []
    list_states = [state]

    for epoch in range(10):
        list_yhats = []
        list_states = [state]
        optimizer.zero_grad()
        for time_step in range(T - 1):
            y_hat, next_state, _ = cell(
                state=list_states[time_step],
                teacher_forcing=True,
                observation=tens_ytrues[time_step]
            )
            list_yhats.append(y_hat)
            list_states.append(next_state)

        # Compute y_hat for the last time step
        last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
        list_yhats.append(last_y_hat)

        # Stack and compute loss
        tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
        loss = loss_fct(tens_yhats, tens_ytrues)
        loss.backward()
        optimizer.step()
        print(f" Printing below the gradient of the D matrix : {cell.D.weight.grad}")
        print (" ====================")
        # print(f" Printing below the gradient of the A matrix : {cell.A.weight.grad}")
        # Assertions
        assert state.grad is not None, "Gradients are not flowing through the initial state."
        assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
        assert cell.D.weight.grad is not None, "Gradients are not flowing through the `DiagonalMatrix` weights."
        assert loss.item() > 0, "Loss should be a positive scalar."