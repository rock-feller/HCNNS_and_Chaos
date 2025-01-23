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
        


        
class partial_teacher_forcing(nn.Dropout):
    """
    Implements dropout with scaling to enable partial teacher forcing.

    This layer applies dropout to the delta term (`y_hat - y_true`) during state transitions
    to simulate partial teacher forcing, allowing for a controlled level of noise or guidance.

    Attributes
    ----------
    p : float
        Dropout probability. The fraction of elements to drop.
    inplace : bool
        Whether to perform the operation in-place.

    Methods
    -------
    forward(input: torch.Tensor) -> torch.Tensor:
        Applies dropout with scaling to the input tensor.
    """
    def __init__(self, p: float = 0., inplace: bool = False):
        """
        Initializes the partial teacher forcing dropout layer.

        Parameters
        ----------
        p : float, optional (default=0.)
            Dropout probability. Defaults to 0 (no dropout).
        inplace : bool, optional (default=False)
            Whether to perform the operation in-place.
        """
        super(partial_teacher_forcing, self).__init__(p, inplace)

        self.p  =  p
        self.inplace = inplace


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies dropout to the input tensor with scaling.

        Parameters
        ----------
        input : torch.Tensor
            The input tensor to which dropout will be applied.

        Returns
        -------
        torch.Tensor
            The scaled tensor with elements randomly dropped based on the dropout probability.
        """
        if not self.training or self.p == 0: #if no training or p = 0
            return input
        else:
            scaled_output = super().forward(input)
            return (1 - self.p) * scaled_output #applies prob dropout without inplace

        


class ptf_cell(nn.Module):
    """
    Partial Teacher Forcing HCNN Cell.

    This class implements the HCNN Cell with support for partial teacher forcing during training.
    It allows a controlled adjustment of the teacher forcing behavior using dropout probabilities, 
    which can simulate the effects of partial guidance in state transitions.

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
    ptf_dropout(prob: float) -> nn.Module:
        Returns a dropout module to apply partial teacher forcing.
    forward(state: torch.Tensor, teacher_forcing: bool, prob: float, 
            observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Performs a forward pass with optional partial teacher forcing during state transitions.
    """


    def __init__(self, n_obs: int, 
                 n_hid_vars: int , 
                 init_range: Tuple[float,float] = (-0.75,0.75)):

        super(ptf_cell, self).__init__()
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

    def ptf_dropout(self, prob: float) -> nn.Module:
        """
        Creates a dropout module for applying partial teacher forcing.

        Parameters
        ----------
        prob : float
            Dropout probability for partial teacher forcing. Determines the fraction of elements in the
            delta term tensor (`y_hat - y_true`) that are randomly dropped (set to 0).

        Returns
        -------
        nn.Module
            A `partial_teacher_forcing` dropout module initialized with the given probability.
        """
        return partial_teacher_forcing(p=prob )


    def forward(self, state: torch.Tensor, teacher_forcing: bool, 
                prob: Optional[float] =  None, 
                observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],Optional[torch.Tensor]]:
        """
        Forward pass of the Partial Teacher Forcing HCNN Cell.

        Parameters
        ----------
        state : torch.Tensor, shape=(n_hid_vars,)
            The current state tensor (`s_t`) of the HCNN, with shape `(n_hid_vars,)`.
        teacher_forcing : bool
            Whether to apply teacher forcing:
            - True: Use the provided observation (`observation`) with optional dropout.
            - False: Compute the next state based solely on the model's prediction.
        prob : float
            Dropout probability for partial teacher forcing. Only used if `teacher_forcing` is True.
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
        - When `teacher_forcing` is True, the delta term is passed through partial dropout before
        being used to compute the next state.
        - Without teacher forcing, the next state is computed purely from the model's prediction.
        """

    # Compute the expected output (y_hat)
        expectation = torch.matmul(self.ConMat, state)

        print("expectation requires_grad:", expectation.requires_grad)
        if teacher_forcing:

            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")

            # Compute the delta term (y_true - y_hat)
            delta_term = observation - expectation

            dropout_deltaterm =  self.ptf_dropout(prob = prob)(delta_term)

            # Teacher forcing: Correct the state using the delta term
            teach_forc = torch.matmul(self.ConMat.T, dropout_deltaterm)

            r_state = state - teach_forc
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, delta_term , dropout_deltaterm , state, r_state
        else:
            # Without teacher forcing: State evolves independently
            r_state = torch.matmul(self.Ide, state)
            next_state = self.A(torch.tanh(r_state))


            return expectation, next_state, None , None , None, None
        

def test_ptf_cell_initialization():
    """Test if the PTFCell initializes correctly."""
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Check attributes
    assert cell.n_obs == n_obs, "Number of observed variables is incorrect."
    assert cell.n_hid_vars == n_hid_vars, "Number of hidden variables is incorrect."
    assert cell.ConMat.shape == (n_obs, n_hid_vars), "Connection matrix shape is incorrect."
    assert cell.Ide.shape == (n_hid_vars, n_hid_vars), "Identity matrix shape is incorrect."
    assert isinstance(cell.A, CustomLinear), "CustomLinear layer is not initialized correctly."


def test_ptf_cell_device_assignment():
    """Test if the default device is assigned correctly."""
    cell = ptf_cell(5, 10)
    expected_device = cell._get_default_device()
    assert cell.device == expected_device, "Device assignment is incorrect."


def test_ptf_cell_forward_no_dropout():
    """Test the forward method without dropout (prob = 0)."""
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars,requires_grad=True)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation
    prob = 0.0  # No dropout

    expectation, next_state, delta_term , partial_delta_term, s_state, r_state= cell.forward(
        state=state, teacher_forcing=True, prob=prob, observation=observation
    )

    # Check shapes of outputs
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert s_state.shape == (n_hid_vars,), "s state shape is incorrect."
    assert r_state.shape == (n_hid_vars,), "r state shape is incorrect."
    assert delta_term.shape == (n_obs,), "Delta term shape is incorrect."
    assert partial_delta_term.shape == (n_obs,), "Partial Delta term shape is incorrect."
    assert torch.allclose(delta_term, observation - expectation), "Delta term is incorrect."


def test_ptf_cell_partial_dropout():
    """Test the forward method with partial teacher forcing (prob > 0)."""
    n_obs, n_hid_vars = 50, 100
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation
    prob = 1. # 50% dropout

    expectation, next_state, delta_term, partial_delta_term , r, s = cell.forward(
        state=state, teacher_forcing=True, prob=prob, observation=observation
    )

    # Check shapes of outputs
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert delta_term.shape == (n_obs,), "Delta term shape is incorrect."
    assert torch.equal(r, s) == True , "The dropout is not applied"
    # Ensure some elements in the delta term are dropped
    dropped_elements = torch.sum(partial_delta_term == 0)
    assert dropped_elements > 0, "Dropout is not applied to delta term as expected."


def test_ptf_cell_with_no_teacher_forcing_no_prob():
    """Test the forward method with partial teacher forcing (prob > 0)."""
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation
    prob = 0.5  # 50% dropout

    expectation, next_state, delta_term, partial_delta_term, s , r = cell.forward(
        state=state, teacher_forcing=False)

    assert delta_term is None, "Delta term should be None when teacher forcing is False."
    assert partial_delta_term is None, "Partial delta term should be None when teacher forcing is False."
    assert s is None, "s should not have a value."
    assert r is None, "r should not have a value."




def test_ptf_cell_consistency_no_dropout():
    """Test if the outputs are consistent when dropout is disabled."""
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation
    prob = 0.0  # No dropout

    output_1 = cell.forward(state, teacher_forcing=True, prob=prob, observation=observation)
    output_2 = cell.forward(state, teacher_forcing=True, prob=prob, observation=observation)

    for o1, o2 in zip(output_1, output_2):
        if o1 is not None:
            assert torch.allclose(o1, o2), "Outputs are inconsistent when dropout is disabled."


def test_ptf_cell_gradient_flow():
    """Test if gradients flow properly through the PTFCell."""
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars, requires_grad=True)  # Enable gradient tracking
    observation = torch.randn(n_obs)  # Random observation
    prob = 0.2  # Partial dropout

    expectation, next_state, delta_term, partial_delta_term, s , r = cell.forward(
        state=state, teacher_forcing=True, prob=prob, observation=observation
    )

    loss_fct = torch.nn.MSELoss()
    loss = loss_fct(expectation, observation)
    loss.backward()  # Backpropagation

    assert state.grad is not None, "Gradients are not flowing through the state."
    assert cell.A.weight.grad is None, "Gradients should not flowing through the CustomLinear weights."



def test_ptf_cell_dropout_edge_cases():
    """Test dropout behavior at edge probabilities (0 and 1)."""
    n_obs, n_hid_vars = 5, 10
    cell = ptf_cell(n_obs, n_hid_vars)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation

    # Case: No dropout (prob = 0)
    prob = 0.0
    _, _, delta_term_no_dropout , delta_term_with_dropout , s, r= cell.forward(state, teacher_forcing=True, prob=prob, observation=observation)
    assert torch.allclose(delta_term_no_dropout, observation - torch.matmul(cell.ConMat, state)), \
        "Delta term is incorrect when dropout is disabled."
    assert  torch.allclose(delta_term_with_dropout, observation - torch.matmul(cell.ConMat, state)), \
        "Delta term is should not have zeros values when dropout is disabled."

    # Case: Full dropout (prob = 1)
    prob = 1.0
    _, _, delta_term_no_dropout, delta_term_with_dropout , s, r = cell.forward(state, teacher_forcing=True, prob=prob, observation=observation)
    print("partial term with dropout is ", torch.abs(delta_term_with_dropout))
    assert torch.all(torch.abs(delta_term_no_dropout) != 0), "Delta term should not be affected by dropout when not being applied to it, despite prob = 1."
    assert torch.all(torch.abs(delta_term_with_dropout) == 0), "dropout Delta term should be fully dropped when prob = 1."



def test_ptf_hcnncell_recurrent():
    
    """Test vanilla_hcnncell in a recurrent fashion over a sequence."""

    print("""This code will generate three things 
        -The computational graph of the Vanilla HCNN Cell applied to a 10-time steps sequence of observations.
        -The gradients of the loss with respect to the initial state s0 and the A weight.""")
    # Define test parameters
    T, n_obs, n_hid_vars = 5, 10, 6  # Number of timesteps, observed vars, hidden vars
    cell = ptf_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

    # Initialize input sequence (T, n_obs)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
    loss_fct = torch.nn.MSELoss()

    print( " ========== Prior to Sequence processing ==========")
    print("Initial state:", state)
    print("Initial A weight:", cell.A.weight)
    # Use lists instead of pre-allocated tensors
    list_yhats = []        # To store y_hat for each time step
    list_delta_terms = []  # To store delta terms for each time step
    list_states = [state]  # To store states (initialize with the first state)
    prob = 0
    # Forward pass over the sequence
    for time_step in range(T - 1):
        y_hat, next_state, delta_term, dropout_deltaterm , s, r  = cell.forward(
            state=list_states[time_step],
            teacher_forcing=True , prob=prob ,
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
    # Alternative loss using delta terms (uncomment if needed)
    # loss = loss_fct(tens_delta_terms, torch.zeros_like(tens_delta_terms))

    # Backward pass
    loss.backward()

    # Print gradients
    print( " ========== Gradient with respect to nodels  parameters ==========")

    print("State gradient:", state.grad)
    assert state.grad is not None, "Gradients are not flowing through the state."


    print("A weight gradient:", cell.A.weight.grad)
    assert cell.A.weight.grad is not None, "Gradients  are not flowing through the CustomLinear weights."