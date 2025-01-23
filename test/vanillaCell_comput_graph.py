
from torchviz import make_dot
from modules import vanilla_cell
import torch
from torch.nn import MSELoss


"""Test vanilla_hcnncell in a recurrent fashion over a sequence."""

print("""This code will generate three things 
      -The computational graph of the Vanilla HCNN Cell applied to a 10-time steps sequence of observations.
      -The gradients of the loss with respect to the initial state s0 and the A weight.""")
# Define test parameters
T, n_obs, n_hid_vars = 5, 4, 6  # Number of timesteps, observed vars, hidden vars
cell = vanilla_cell(n_obs=n_obs, n_hid_vars=n_hid_vars)

# Initialize input sequence (T, n_obs)
tens_ytrues = torch.randn(T, n_obs)  # Random sequence of observations
state = torch.randn(n_hid_vars, requires_grad=True)  # Initial hidden state
loss_fct = MSELoss()

print( " ========== Prior to Sequence processing ==========")
print("Initial state:", state)
print("Initial A weight:", cell.A.weight)
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
# Alternative loss using delta terms (uncomment if needed)
# loss = loss_fct(tens_delta_terms, torch.zeros_like(tens_delta_terms))

# Backward pass
loss.backward()

# Print gradients
print( " ========== Gradient with respect to nodels  parameters ==========")
print("State gradient:", state.grad)
print("A weight gradient:", cell.A.weight.grad)

graph = make_dot(loss, params = {"A": cell.A.weight,
                            "ConMat": cell.ConMat, "Ide": cell.Ide,
                            "loss term": loss, "state_s0": state})

graph.render("computational_graph_HCNN_Arch", format="png")