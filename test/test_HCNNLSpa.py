import pytest
import torch
from torch.nn import MSELoss
from typing import Tuple, Optional
import torch.nn as nn



class CustomSparseLinear(nn.Linear):
    def __init__(self, n_hid_vars: int, 
                 bias: bool = False,
                 init_range: Tuple[float, float] = (-0.75, 0.75),
                 sparsity: float = 0.0,
                 mask_type: str = "random",
                 p: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        CustomLinear layer with sparsity applied based on the mask type.

        Parameters
        ----------
        n_hid_vars : int
            Number of hidden variables (both input and output features).
        bias : bool, optional
            If True, includes a bias term. Default is False.
        init_range : Tuple[float, float], optional
            Range for uniform initialization of weights. Default is (-0.75, 0.75).
        sparsity : float, optional
            Proportion of weights to set to zero (random sparsity). Default is 0.0 (no sparsity).
        mask_type : str, optional
            Type of mask to apply. Options:
            - "random": Random sparsity over the entire weight matrix.
            - "non_obs_only": Sparsity applied only on the non-observable block of the weight matrix.
        p : int, optional
            Number of observable variables when using `non_obs_only` mask type.
        device : torch.device, optional
            The device to use for computations. If not specified, automatically selects
            `cuda`, `mps`, or `cpu` based on availability.

        Notes
        -----
        - For `random`, the entire weight matrix has a proportion of its elements randomly zeroed out.
        - For `non_obs_only`, sparsity is applied to the lower-right block of the weight matrix
          (size `(n_hid_vars - p, n_hid_vars)`) with `p` observable variables.
        """
        self.device = device if device else self._get_default_device()

        super(CustomSparseLinear, self).__init__(n_hid_vars, n_hid_vars, bias=bias, device=self.device)

        self.init_range = init_range

        """
        Raises
        ------
        ValueError
            If `sparsity` is not between 0 and 1.
        """
        if not (0 <= sparsity <= 1):
            raise ValueError("Sparsity must be between 0 and 1.")
        self.sparsity = sparsity
        self.mask_type = mask_type
        self.p = p
        self.n_hid_vars = n_hid_vars

        # Initialize weights and biases
        self._initialize_weights()

        # Generate the mask based on the mask type
        self.mask = self._generate_mask()

        # Apply the mask to the weights
        self._apply_mask()

        # Register hook to enforce the mask during backpropagation
        self.weight.register_hook(self._enforce_mask)

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

    def _initialize_weights(self):
        """
        Initializes weights and biases within the specified range.
        """
        nn.init.uniform_(self.weight.data, self.init_range[0], self.init_range[1])
        if self.bias is not None:
            nn.init.uniform_(self.bias.data, self.init_range[0], self.init_range[1])

    def _generate_mask(self) -> torch.Tensor:
        """
        Generates the sparsity mask based on the specified `mask_type`.

        Returns
        -------
        torch.Tensor
            Binary mask of shape `(n_hid_vars, n_hid_vars)`.

        Raises
        ------
        ValueError
            If an invalid `mask_type` is provided.
        """
        mask = torch.ones(self.n_hid_vars, self.n_hid_vars, device=self.device)

        if self.mask_type == "random":
            # Random sparsity over the entire weight matrix
            total_weights = self.n_hid_vars * self.n_hid_vars
            zeroed_weights = int(self.sparsity * total_weights)
            random_indices = torch.randperm(total_weights, device=self.device)[:zeroed_weights]
            flat_mask = mask.view(-1)
            flat_mask[random_indices] = 0.0
            mask = flat_mask.view(self.n_hid_vars, self.n_hid_vars)

        elif self.mask_type == "non_obs_only":
            if self.p is None or not (0 < self.p < self.n_hid_vars):
                raise ValueError("`p` must be provided and satisfy 0 < p < n_hid_vars for `non_obs_only`.")

            # Split the weight matrix into two blocks
            obs_block = torch.ones((self.n_hid_vars, self.p), device=self.device)  # (n_hid_vars, p)
            non_obs_block = torch.ones((self.n_hid_vars, self.n_hid_vars - self.p), device=self.device)  # (n_hid_vars, n_hid_vars - p)

            # Apply sparsity only on the non-observable block
            total_weights_non_obs = non_obs_block.numel()
            zeroed_weights = int(self.sparsity * total_weights_non_obs)
            random_indices = torch.randperm(total_weights_non_obs, device=self.device)[:zeroed_weights]
            flat_non_obs = non_obs_block.view(-1)
            flat_non_obs[random_indices] = 0.0
            non_obs_block = flat_non_obs.view(self.n_hid_vars, self.n_hid_vars - self.p)

            # Combine the blocks to form the mask
            mask = torch.cat((obs_block, non_obs_block), dim=1)

        else:
            raise ValueError(f"Invalid mask_type: {self.mask_type}. Must be 'random' or 'non_obs_only'.")

        return mask

    def _apply_mask(self):
        """
        Applies the mask to the weight matrix.
        """
        self.weight.data *= self.mask

    def _enforce_mask(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Hook to enforce the mask during backpropagation.

        Ensures that:
        - The zeroed-out weights remain zeroed and are not updated.
        - The gradients corresponding to zeroed-out weights are also set to zero.

        Parameters
        ----------
        grad : torch.Tensor
            Gradient of the loss with respect to the weight matrix.

        Returns
        -------
        torch.Tensor
            Modified gradient respecting the mask.
        """
        with torch.no_grad():
            # Enforce the mask on the weights
            self.weight.data *= self.mask
        # Zero out gradients for masked weights
        return grad * self.mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomLinear layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_hid_vars)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_hid_vars)`.
        """
        return nn.functional.linear(input, self.weight, self.bias)
class LargeSparse_cell(nn.Module):
    """
    Large Sparse HCNN Cell.

    This class implements the Large Sparse HCNN Cell for state-based predictions and state transitions.
    It supports the use of teacher forcing during training and includes a sparsity-enabled linear transformation
    for efficiently handling large state spaces.

    Attributes
    ----------
    n_obs : int
        Number of observed variables (output dimension).
    n_hid_vars : int
        Number of hidden variables (state dimension).
    Sparse_A : CustomSparseLinear
        Sparse linear transformation module for updating the hidden state.
    ConMat : torch.Tensor
        Connection matrix used for mapping hidden states to observations.
    Ide : torch.Tensor
        Identity matrix used in internal computations.
    device : torch.device
        The device (CPU, CUDA, or MPS) where the model and tensors are stored.

    Methods
    -------
    forward(state: torch.Tensor, teacher_forcing: bool, observation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Performs a forward pass through the Large Sparse HCNN Cell, computing predictions (`expectation`) 
        and updating the internal state (`next_state`).
    """

    def __init__(self, n_obs: int, 
                 n_hid_vars: int, 
                 init_range: Tuple[float, float] = (-0.75, 0.75),
                 sparsity: float = 0.0,
                 mask_type: str = "random",
                 p: Optional[int] = None):
        """
        Initializes the Large Sparse HCNN Cell with a sparsity-enabled CustomSparseLinear.

        Parameters
        ----------
        n_obs : int
            Number of observed variables (output dimension).
        n_hid_vars : int
            Number of hidden variables (state dimension).
        init_range : Tuple[float, float], optional
            Range for uniform initialization of weights in CustomSparseLinear. Default is (-0.75, 0.75).
        sparsity : float, optional
            Proportion of weights to set to zero (random sparsity). Default is 0.0 (no sparsity).
        mask_type : str, optional
            Type of sparsity mask: "random" or "non_obs_only". Default is "random".
        p : int, optional
            Number of observable variables for "non_obs_only" mask type. Required if `mask_type` is "non_obs_only".
        """
        super(LargeSparse_cell, self).__init__()
        self.n_obs = n_obs
        self.n_hid_vars = n_hid_vars

        # Initialize the sparse CustomSparseLinear layer
        self.Sparse_A = CustomSparseLinear(
            n_hid_vars=n_hid_vars,
            bias=False,
            init_range=init_range,
            sparsity=sparsity,
            mask_type=mask_type,
            p=p
        )

        # Register connection and identity matrices
        self.register_buffer(name='ConMat', tensor=torch.eye(n_obs, n_hid_vars), persistent=False)
        self.register_buffer(name='Ide', tensor=torch.eye(n_hid_vars), persistent=False)

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
        Forward pass of the Large Sparse HCNN Cell.

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
            - Apply a non-linear transformation using the `CustomSparseLinear` layer (`Sparse_A`).
        """
        # Compute the expected output (y_hat)
        expectation = torch.matmul(self.ConMat, state)

        if teacher_forcing:
            if observation is None:
                raise ValueError("`observation` must be provided when `teacher_forcing` is True.")

            # Compute the delta term (y_true - y_hat)
            delta_term = observation - expectation

            # Teacher forcing: Correct the state using the delta term
            teach_forc = torch.matmul(self.ConMat.T, delta_term)

            # Residual state and next state
            r_state = state - teach_forc
            next_state = self.Sparse_A(torch.tanh(r_state))

            return expectation, next_state, delta_term
        else:
            # Without teacher forcing: State evolves independently
            r_state = torch.matmul(self.Ide, state)
            next_state = self.Sparse_A(torch.tanh(r_state))

            return expectation, next_state, None

def test_initialization():
    """Test if the LargeSparse_cell initializes correctly."""
    n_obs = 4
    n_hid_vars = 6
    sparsity = 0.5
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity)

    # Check attributes
    assert cell.n_obs == n_obs, "Number of observed variables is incorrect."
    assert cell.n_hid_vars == n_hid_vars, "Number of hidden variables is incorrect."
    assert cell.ConMat.shape == (n_obs, n_hid_vars), "Connection matrix shape is incorrect."
    assert cell.Ide.shape == (n_hid_vars, n_hid_vars), "Identity matrix shape is incorrect."
    assert isinstance(cell.Sparse_A, CustomSparseLinear), "Sparse_A is not properly initialized."


def test_device_assignment():
    """Test if the default device is assigned correctly."""
    cell = LargeSparse_cell(4, 6)
    expected_device = cell._get_default_device()
    assert cell.device == expected_device, "Device assignment is incorrect."


def test_sparsity_random_mask():
    """Test if random sparsity mask is applied correctly in the CustomSparseLinear layer."""
    n_obs = 4
    n_hid_vars = 6
    sparsity = 0.3
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity, mask_type="random")

    mask = cell.Sparse_A.mask
    assert mask is not None, "Mask should be created for random sparsity."
    num_weights = n_hid_vars * n_hid_vars
    num_zeros = torch.sum(mask == 0).item()
    expected_zeros = int(num_weights * sparsity)
    assert num_zeros == expected_zeros, "Incorrect number of zeroed weights in the mask."


def test_sparsity_non_obs_only():
    """Test if non_obs_only mask is applied correctly."""
    n_obs = 3
    n_hid_vars = 5
    sparsity = 0.4
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity, mask_type="non_obs_only", p=n_obs)

    mask = cell.Sparse_A.mask
    assert mask is not None, "Mask should be created for non_obs_only."
    assert mask[:, :n_obs].sum() == n_hid_vars * n_obs, "Observable part of the mask should be dense."
    hidden_mask = mask[:, n_obs:]
    num_zeros = torch.sum(hidden_mask == 0).item()
    num_weights = hidden_mask.numel()
    expected_zeros = int(num_weights * sparsity)
    assert num_zeros == expected_zeros, "Incorrect number of zeroed weights in the hidden part of the mask."


def test_forward_no_teacher_forcing():
    """Test the forward pass without teacher forcing."""
    n_obs, n_hid_vars = 4, 6
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.3)

    state = torch.randn(n_hid_vars)  # Random initial state
    expectation, next_state, delta_term = cell(state, teacher_forcing=False)

    # Check output shapes
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert delta_term is None, "Delta term should be None when teacher forcing is False."


def test_forward_with_teacher_forcing():
    """Test the forward pass with teacher forcing."""
    n_obs, n_hid_vars = 4, 6
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.2)

    state = torch.randn(n_hid_vars)  # Random initial state
    observation = torch.randn(n_obs)  # Random observation
    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    # Check output shapes
    assert expectation.shape == (n_obs,), "Output expectation shape is incorrect."
    assert next_state.shape == (n_hid_vars,), "Next state shape is incorrect."
    assert delta_term.shape == (n_obs,), "Delta term shape is incorrect."


def test_teacher_forcing_without_observation():
    """Test if an error is raised when teacher forcing is enabled but observation is missing."""
    n_obs, n_hid_vars = 4, 6
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.3)

    state = torch.randn(n_hid_vars)  # Random initial state

    with pytest.raises(ValueError, match="`observation` must be provided when `teacher_forcing` is True."):
        _ = cell(state, teacher_forcing=True)


def test_gradient_flow():
    """Test if gradients flow properly through the model."""
    n_obs, n_hid_vars = 4, 6
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.1)

    state = torch.randn(n_hid_vars, requires_grad=True)  # Enable gradient tracking
    some_target_state = torch.randn(n_hid_vars)  # Random target state
    observation = torch.randn(n_obs)  # Random observation

    expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)

    loss_fct = MSELoss()
    loss = loss_fct(next_state, some_target_state)
    loss.backward()  # Backpropagation

    assert state.grad is not None, "Gradients are not flowing through the initial state."
    assert cell.Sparse_A.weight.grad is not None, "Gradients are not flowing through the sparse weights."
    assert torch.allclose(cell.Sparse_A.weight.grad * cell.Sparse_A.mask, cell.Sparse_A.weight.grad), \
        "Gradients for masked weights should be zero."


def test_sparsity_retention_during_training():
    """Test that the sparsity mask is retained during weight updates."""
    n_obs, n_hid_vars = 4, 6
    sparsity = 0.2
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity)

    optimizer = torch.optim.SGD(cell.parameters(), lr=0.01)

    for _ in range(5):
        state = torch.randn(n_hid_vars, requires_grad=True)
        observation = torch.randn(n_obs)

        expectation, next_state, delta_term = cell(state, teacher_forcing=True, observation=observation)
        loss_fct = MSELoss()
        loss = loss_fct(expectation, observation)
        loss.backward()
        optimizer.step()

        # Check that the sparsity mask is retained
        with torch.no_grad():
            assert torch.allclose(cell.Sparse_A.weight * cell.Sparse_A.mask, cell.Sparse_A.weight), \
                "Sparsity mask is not retained during training."

def test_recurrent_behavior_largesparse():
    """Test the Large Sparse HCNN Cell in a recurrent setup over a sequence."""
    T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.3)

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
    assert cell.Sparse_A.weight.grad is not None, "Gradients are not flowing through the sparse weights."
    assert loss.item() > 0, "Loss should be a positive scalar."
    assert torch.allclose(cell.Sparse_A.weight.grad * cell.Sparse_A.mask, cell.Sparse_A.weight.grad), \
        "Gradients for masked weights should be zero."


def test_recurrent_behavior_largesparse_across_epochs():
    """Test the Large Sparse HCNN Cell in a recurrent setup over multiple epochs."""
    T, n_obs, n_hid_vars = 3, 4, 6  # Number of timesteps, observed vars, hidden vars
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=0.2, mask_type="non_obs_only",p=n_obs)

    # Initialize input sequence (T, n_obs)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
    state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
    loss_fct = MSELoss()
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.01)

    for epoch in range(5):  # Test over 5 epochs
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

        print(f"Epoch {epoch + 1}:")
        print(f"Sparse_A weight gradients:\n{cell.Sparse_A.weight.grad}")
        print(f"Sparse_A weights after update:\n{cell.Sparse_A.weight}")
        print("=" * 50)

        # Assertions
        assert state.grad is not None, "Gradients are not flowing through the initial state."
        assert cell.Sparse_A.weight.grad is not None, "Gradients are not flowing through the sparse weights."
        assert torch.allclose(cell.Sparse_A.weight.grad * cell.Sparse_A.mask, cell.Sparse_A.weight.grad), \
            "Gradients for masked weights should be zero."
        assert loss.item() > 0, "Loss should be a positive scalar."


def test_mask_retention_recurrent_training():
    """Test that the sparsity mask is retained during recurrent training."""
    T, n_obs, n_hid_vars = 3, 4, 6
    sparsity = 0.3
    cell = LargeSparse_cell(n_obs=n_obs, n_hid_vars=n_hid_vars, sparsity=sparsity, mask_type="non_obs_only", p=n_obs)

    optimizer = torch.optim.Adam(cell.parameters(), lr=0.01)
    tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations

    for epoch in range(3):  # Test for 3 epochs
        state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
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

        # Compute loss and update
        tens_yhats = torch.stack(list_yhats)
        loss = MSELoss()(tens_yhats, tens_ytrues)
        loss.backward()
        optimizer.step()

        # Ensure the sparsity mask is retained
        with torch.no_grad():
            assert torch.allclose(cell.Sparse_A.weight * cell.Sparse_A.mask, cell.Sparse_A.weight), \
                "Sparsity mask is not retained during recurrent training."


# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__])

