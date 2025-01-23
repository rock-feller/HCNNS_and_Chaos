import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import MSELoss
import unittest



class DiagonalMatrix(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias=False, init_diag: Optional[float] = 1.0):
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
            self.weight.data = torch.mul(self.weight.data, self.mask)  # Retain only diagonal elements
            self.weight.data.clamp_(min=0., max=1)

        return grad * self.mask  # Mask gradients to zero out off-diagonal elements

    def forward(self, input, verbose: bool = False):
        if verbose:
            print(f"Diagonal weights: {torch.diag(self.weight)}")
        return nn.functional.linear(input, self.weight, self.bias)


def test_gradient_hook_applied():
    """
    Test that the gradient of the diagonal matrix has off-diagonal elements zeroed out
    when the hook is applied.
    """
    in_features = 5
    matrix = DiagonalMatrix(in_features, in_features, bias=False)

    # Create input and target tensors
    input_tensor = torch.randn(10, in_features)  # Batch of inputs
    target = torch.randn(10, in_features)  # Target values

    # Define a simple loss
    output = matrix(input_tensor)
    loss = MSELoss()(output, target)

    # Perform backward pass
    loss.backward()

    # Check that gradients for off-diagonal elements are zero
    grad = matrix.weight.grad
    assert grad is not None, "Gradient was not computed."
    off_diagonal_mask = ~torch.eye(in_features, dtype=torch.bool)
    assert torch.all(grad[off_diagonal_mask] == 0), "Off-diagonal gradient elements are not zero."
    print("Gradient hook applied correctly: off-diagonal elements zeroed out.")


def test_only_diagonal_retained_without_gradients():
    """
    Test that when gradients are not computed, only the diagonal elements are retained in the weight matrix.
    """
    in_features = 5
    matrix = DiagonalMatrix(in_features, in_features, bias=False)

    # Modify the weight matrix to add non-diagonal elements
    matrix.weight.data += torch.randn_like(matrix.weight.data)  # Add random values

    # Check that the weight matrix has off-diagonal elements initially
    off_diagonal_mask = ~torch.eye(in_features, dtype=torch.bool)
    assert torch.any(matrix.weight.data[off_diagonal_mask] != 0), \
        "Weight matrix off-diagonal elements are already zero."

    # Trigger the hook without gradients by manually calling `_clamp_and_zero_out`
    _ = matrix._clamp_and_zero_out(torch.zeros_like(matrix.weight))  # Simulate the hook

    # Check that only diagonal elements are retained
    assert torch.all(matrix.weight.data[off_diagonal_mask] == 0), \
        "Off-diagonal elements are not zeroed out after the hook."
    print("Only diagonal elements retained without gradients.")

    # Check diagonal elements are untouched
    diagonal_mask = torch.eye(in_features, dtype=torch.bool)
    assert torch.all(matrix.weight.data[diagonal_mask] >= 0), \
        "Diagonal elements have been incorrectly modified."


def test_diagonal_matrix_training():
    """
    Test that the DiagonalMatrix updates only its diagonal elements during training, and off-diagonal
    values are clamped to zero after each weight update.
    """
    in_features = 5
    num_epochs = 100
    learning_rate = 0.01

    # Initialize the DiagonalMatrix
    matrix = DiagonalMatrix(in_features, in_features, bias=False)

    # Create input and target tensors
    input_tensor = torch.randn(10, in_features)  # Batch of inputs
    target = torch.randn(10, in_features)  # Target values

    # Set up optimizer and loss function
    optimizer = torch.optim.SGD(matrix.parameters(), lr=learning_rate)
    loss_function = MSELoss()

    # Get the off-diagonal mask
    off_diagonal_mask = ~torch.eye(in_features, dtype=torch.bool)

    for epoch in range(num_epochs):
        # Forward pass
        output = matrix(input_tensor)
        loss = loss_function(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check that off-diagonal gradients are zero
        assert torch.all(matrix.weight.grad[off_diagonal_mask] == 0), \
            f"Off-diagonal gradients are not zero at epoch {epoch + 1}."

        # Weight update
        optimizer.step()
        print (" Print the gradients: " ,  matrix.weight.grad)
        # Verify that off-diagonal weights remain zero after the update
        assert torch.all(matrix.weight.data[off_diagonal_mask] == 0), \
            f"Off-diagonal weights are not clamped to zero at epoch {epoch + 1}."

        # Print progress (optional for debugging)
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Ensure weights were updated (i.e., learning occurred)
    initial_weights = torch.eye(in_features)
    assert not torch.allclose(matrix.weight.data, initial_weights), \
        "Diagonal weights did not update during training."

    print("Test passed: DiagonalMatrix trained successfully and off-diagonal constraints maintained.")


def test_initialization():
    """
    Test that the DiagonalMatrix initializes correctly with the specified diagonal values.
    """
    in_features = 5
    init_diag = 0.5
    matrix = DiagonalMatrix(in_features, in_features, bias=False, init_diag=init_diag)

    # Check that the diagonal elements are initialized correctly
    diagonal_mask = torch.eye(in_features, dtype=torch.bool)
    assert torch.all(matrix.weight.data[diagonal_mask] == init_diag), \
        "Diagonal elements are not initialized correctly."
    print("Initialization test passed: Diagonal elements initialized correctly.")


def test_forward_pass():
    """
    Test the forward pass of the DiagonalMatrix to ensure it computes the correct output.
    """
    in_features = 5
    matrix = DiagonalMatrix(in_features, in_features, bias=False)

    # Create input tensor
    input_tensor = torch.randn(10, in_features)  # Batch of inputs

    # Perform forward pass
    output = matrix(input_tensor)

    # Check the output shape
    assert output.shape == (10, in_features), "Output shape is incorrect."
    print("Forward pass test passed: Output shape is correct.")


def test_bias():
    """
    Test that the bias is correctly applied in the forward pass.
    """
    in_features = 5
    matrix = DiagonalMatrix(in_features, in_features, bias=True)

    # Create input tensor
    input_tensor = torch.randn(10, in_features)  # Batch of inputs

    # Perform forward pass
    output = matrix(input_tensor)

    # Check that the bias is applied
    assert matrix.bias is not None, "Bias is not initialized."
    assert torch.allclose(output - matrix.bias, nn.functional.linear(input_tensor, matrix.weight)), \
        "Bias is not correctly applied in the forward pass."
    print("Bias test passed: Bias is correctly applied in the forward pass.")

    class TestDiagonalMatrix(unittest.TestCase):

        def test_gradient_hook_applied(self):
            in_features = 5
            matrix = DiagonalMatrix(in_features, in_features, bias=False)
            input_tensor = torch.randn(10, in_features)
            target = torch.randn(10, in_features)
            output = matrix(input_tensor)
            loss = MSELoss()(output, target)
            loss.backward()
            grad = matrix.weight.grad
            self.assertIsNotNone(grad, "Gradient was not computed.")
            off_diagonal_mask = ~torch.eye(in_features, dtype=torch.bool)
            self.assertTrue(torch.all(grad[off_diagonal_mask] == 0), "Off-diagonal gradient elements are not zero.")

        def test_only_diagonal_retained_without_gradients(self):
            in_features = 5
            matrix = DiagonalMatrix(in_features, in_features, bias=False)
            matrix.weight.data += torch.randn_like(matrix.weight.data)
            off_diagonal_mask = ~torch.eye(in_features, dtype=torch.bool)
            self.assertTrue(torch.any(matrix.weight.data[off_diagonal_mask] != 0), "Weight matrix off-diagonal elements are already zero.")
            _ = matrix._clamp_and_zero_out(torch.zeros_like(matrix.weight))
            self.assertTrue(torch.all(matrix.weight.data[off_diagonal_mask] == 0), "Off-diagonal elements are not zeroed out after the hook.")
            diagonal_mask = torch.eye(in_features, dtype=torch.bool)
            self.assertTrue(torch.all(matrix.weight.data[diagonal_mask] >= 0), "Diagonal elements have been incorrectly modified.")

        def test_diagonal_matrix_training(self):
            in_features = 5
            num_epochs = 100
            learning_rate = 0.01
            matrix = DiagonalMatrix(in_features, in_features, bias=False)
            input_tensor = torch.randn(10, in_features)
            target = torch.randn(10, in_features)
            optimizer = torch.optim.SGD(matrix.parameters(), lr=learning_rate)
            loss_function = MSELoss()
            off_diagonal_mask = ~torch.eye(in_features, dtype=torch.bool)
            for epoch in range(num_epochs):
                output = matrix(input_tensor)
                loss = loss_function(output, target)
                optimizer.zero_grad()
                loss.backward()
                self.assertTrue(torch.all(matrix.weight.grad[off_diagonal_mask] == 0), f"Off-diagonal gradients are not zero at epoch {epoch + 1}.")
                optimizer.step()
                self.assertTrue(torch.all(matrix.weight.data[off_diagonal_mask] == 0), f"Off-diagonal weights are not clamped to zero at epoch {epoch + 1}.")
            initial_weights = torch.eye(in_features)
            self.assertFalse(torch.allclose(matrix.weight.data, initial_weights), "Diagonal weights did not update during training.")

        def test_initialization(self):
            in_features = 5
            init_diag = 0.5
            matrix = DiagonalMatrix(in_features, in_features, bias=False, init_diag=init_diag)
            diagonal_mask = torch.eye(in_features, dtype=torch.bool)
            self.assertTrue(torch.all(matrix.weight.data[diagonal_mask] == init_diag), "Diagonal elements are not initialized correctly.")

        def test_forward_pass(self):
            in_features = 5
            matrix = DiagonalMatrix(in_features, in_features, bias=False)
            input_tensor = torch.randn(10, in_features)
            output = matrix(input_tensor)
            self.assertEqual(output.shape, (10, in_features), "Output shape is incorrect.")

        def test_bias(self):
            in_features = 5
            matrix = DiagonalMatrix(in_features, in_features, bias=True)
            input_tensor = torch.randn(10, in_features)
            output = matrix(input_tensor)
            self.assertIsNotNone(matrix.bias, "Bias is not initialized.")
            self.assertTrue(torch.allclose(output - matrix.bias, nn.functional.linear(input_tensor, matrix.weight)), "Bias is not correctly applied in the forward pass.")

    if __name__ == "__main__":
        unittest.main()
