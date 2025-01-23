
from torchviz import make_dot
from modules import lstm_cell
import torch
from torch.nn import MSELoss


"""Test vanilla_hcnncell in a recurrent fashion over a sequence."""

print("""This code will generate three things 
      -The computational graph of the  HCNNLForm Cell applied to a 10-time steps sequence of observations.
      -The gradients of the loss with respect to the initial state s0 and the A weight.""")
# Define test parameters
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

# for epoch in range(10):
list_yhats = []
list_delta_terms = []
list_states = [state]
optimizer.zero_grad()
for time_step in range(T - 1):
    y_hat, next_state, delta_term = cell(
        state=list_states[time_step],
        teacher_forcing=True,
        observation=tens_ytrues[time_step]
    )
    list_yhats.append(y_hat)
    list_delta_terms.append(delta_term)
    list_states.append(next_state)

# Compute y_hat for the last time step
last_y_hat = torch.matmul(cell.ConMat, list_states[-1])
list_yhats.append(last_y_hat)
# list_delta_terms.append(last_delta_term)

# Stack and compute loss
tens_yhats = torch.stack(list_yhats)  # Shape: (T, n_obs)
loss = loss_fct(tens_yhats, tens_ytrues)
loss.backward()
    # print(f" Printing below the gradient of the A matrix : {cell.A.weight.grad}")
    # Assertions
    # assert state.grad is not None, "Gradients are not flowing through the initial state."
    # assert cell.A.weight.grad is not None, "Gradients are not flowing through the `CustomLinear` weights."
    # assert cell.D.weight.grad is not None, "Gradients are not flowing through the `DiagonalMatrix` weights."
    # assert loss.item() > 0, "Loss should be a positive scalar."
# Print gradients
print( " ========== Gradient with respect to nodels  parameters ==========")
print("State gradient:", state.grad)
print("A weight gradient:", cell.A.weight.grad)
print(f" Printing below the gradient of the D matrix : {cell.D.weight.grad}")

graph = make_dot(loss, params = {"A": cell.A.weight, "D": cell.D.weight, 
                            "ConMat": cell.ConMat, "Ide": cell.Ide,
                            "loss term": loss, "state_s0": state})

graph.render("computational_graph_HCNNLForm_Arch", format="png")
