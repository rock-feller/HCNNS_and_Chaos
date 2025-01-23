import torch
import pytest
from torch.nn import MSELoss
from typing import Tuple
# from custom_linear import CustomLinear  # Import the updated CustomLinear class
import torch
import torch.nn as nn
from typing import Optional, Tuple


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


def test_weight_initialization():
    """Test that the weight matrix is initialized within the specified range."""
    n_hid_vars = 5
    init_range = (-0.5, 0.5)
    layer = CustomSparseLinear(n_hid_vars, init_range=init_range)

    assert torch.all(layer.weight.data >= init_range[0]), "Weights initialized below minimum range."
    assert torch.all(layer.weight.data <= init_range[1]), "Weights initialized above maximum range."

def test_device_assignment():
    """Test that the device is correctly assigned."""
    n_hid_vars = 5
    layer = CustomSparseLinear(n_hid_vars)

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert layer.weight.device.type == expected_device, f"Expected device: {expected_device}, but got {layer.weight.device.type}"

def test_random_sparsity_mask():
    """Test the creation of a random sparsity mask."""
    n_hid_vars = 5
    sparsity = 0.4
    layer = CustomSparseLinear(n_hid_vars, sparsity=sparsity, mask_type="random")

    # Calculate the number of zeroed weights
    total_weights = n_hid_vars * n_hid_vars
    zeroed_weights = int(sparsity * total_weights)
    actual_zeroed = (layer.mask == 0).sum().item()

    assert layer.mask.shape == (n_hid_vars, n_hid_vars), "Mask shape is incorrect."
    assert actual_zeroed == zeroed_weights, f"Expected {zeroed_weights} zeroed weights, but got {actual_zeroed}."

def test_non_obs_only_mask():
    """Test the creation of a non-obs-only sparsity mask."""
    n_hid_vars = 6
    p = 2
    sparsity = 0.5
    layer = CustomSparseLinear(n_hid_vars, sparsity=sparsity, mask_type="non_obs_only", p=p)

    # Check the shape of the mask
    assert layer.mask.shape == (n_hid_vars, n_hid_vars), "Mask shape is incorrect."

    # Ensure only the non-observable block is sparse
    obs_block = layer.mask[:, :p]
    non_obs_block = layer.mask[:, p:]

    assert torch.all(obs_block == 1), "Observable block contains zeroed weights."
    total_weights_non_obs = non_obs_block.numel()
    expected_zeroed = int(sparsity * total_weights_non_obs)
    actual_zeroed = (non_obs_block == 0).sum().item()
    assert actual_zeroed == expected_zeroed, f"Expected {expected_zeroed} zeroed weights, but got {actual_zeroed}."

def test_enforce_mask_during_forward():
    """Test that the mask is correctly applied during the forward pass."""
    n_hid_vars = 4
    sparsity = 0.5
    layer = CustomSparseLinear(n_hid_vars, sparsity=sparsity, mask_type="random")

    # Forward pass
    input_tensor = torch.randn(3, n_hid_vars)
    output = layer(input_tensor)

    # Check that masked weights remain zero
    masked_weights = layer.weight.data * layer.mask
    assert torch.allclose(layer.weight.data, masked_weights), "Masked weights are not being enforced."

def test_gradients_respect_mask():
    """Test that gradients respect the mask during backpropagation."""
    n_hid_vars = 5
    sparsity = 0.4
    layer = CustomSparseLinear(n_hid_vars, sparsity=sparsity, mask_type="random")

    # Forward pass
    input_tensor = torch.randn(10, n_hid_vars)
    target = torch.randn(10, n_hid_vars)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

    output = layer(input_tensor)
    loss = MSELoss()(output, target)
    loss.backward()

    # Ensure gradients respect the mask
    grad_masked = layer.weight.grad * layer.mask
    assert torch.allclose(layer.weight.grad, grad_masked), "Gradients are not respecting the mask."

def test_mask_persistence():
    """Test that the mask persists across backpropagation and updates."""
    n_hid_vars = 5
    sparsity = 0.4
    layer = CustomSparseLinear(n_hid_vars, sparsity=sparsity, mask_type="random")
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

    # Perform multiple training steps
    for _ in range(5):
        input_tensor = torch.randn(10, n_hid_vars)
        target = torch.randn(10, n_hid_vars)
        output = layer(input_tensor)
        loss = MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Ensure the mask is still applied
        masked_weights = layer.weight.data * layer.mask
        assert torch.allclose(layer.weight.data, masked_weights), "Mask is not persisting across updates."

def test_invalid_mask_type():
    """Test that an invalid mask type raises an error."""
    n_hid_vars = 5
    with pytest.raises(ValueError, match="Invalid mask_type"):
        CustomSparseLinear(n_hid_vars, mask_type="invalid")

def test_invalid_non_obs_only_p():
    """Test that an invalid value for `p` raises an error for non-obs-only mask."""
    n_hid_vars = 5
    with pytest.raises(ValueError, match="`p` must be provided and satisfy 0 < p < n_hid_vars"):
        CustomSparseLinear(n_hid_vars, sparsity=0.5, mask_type="non_obs_only", p=10)

def test_sparsity_range():
    """Test that an invalid sparsity value raises an error."""
    n_hid_vars = 5
    
    # Test for sparsity > 1
    with pytest.raises(ValueError, match="Sparsity must be between 0 and 1"):
        CustomSparseLinear(n_hid_vars, sparsity=1.5)

    # Test for sparsity < 0
    with pytest.raises(ValueError, match="Sparsity must be between 0 and 1"):
        CustomSparseLinear(n_hid_vars, sparsity=-0.1)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
