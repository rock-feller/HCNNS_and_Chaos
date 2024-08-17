import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data.chaotic_data import LorenzSolver , RabinovichFabrikantSolver , RosslerSolver
from data_prep.data_utils_fcts import Normalization_Strategy  , Noisification_Strategy, InpTarg_TSDataset , SlidingWindowDataset

from models.rnn_based_models import RNNModel , LSTMModel
from modeling_strategy.causal_modeling import RNNs_modelling_scenario
from utils_fcts import PlottingTool , DataCaster 

PlotTool =  PlottingTool()
DataCast = DataCaster()
# lor_trajs , times =  LorenzSolver(start=0,stop =100, ics=(0,1,1.05) , time_grid = 0.04)
lor_trajs , times =  RosslerSolver(start=0,stop =100, ics=(1,1,1.) , time_grid = 0.04)

norma =  Normalization_Strategy()
lor_trajs_sc , tens_avgs = norma.Scale_ToNormalize(lor_trajs, 0.035)
noisy =  Noisification_Strategy()
noizy_lortrajs =  noisy.adding_sigmanoise_from_normal(lor_trajs_sc , 0.)

forecast_horizon = 500

tensor_traindata =noizy_lortrajs[:-forecast_horizon].squeeze(2)#
tensor_testdata  =noizy_lortrajs[-forecast_horizon:].squeeze(2)#

# Create the dataset and data loader
context_window_size = 200
batch_size = 10

train_loader = InpTarg_TSDataset.create_train_loader(tensor_traindata, context_window_size, batch_size, shuffle=False)


# last_
slided_traindataset = SlidingWindowDataset(noizy_lortrajs.squeeze(2), context_window_size)

context_window_l , true_testdata_l = slided_traindataset.contextwindow_testdata_generator(context_window_size, forecast_horizon,  'last_')

#random_ 
slided_traindataset = SlidingWindowDataset(tensor_traindata, context_window_size) 
context_window_r , true_testdata_r = slided_traindataset.contextwindow_testdata_generator(context_window_size, forecast_horizon,  'random_')

# ========================================================================
# ========================================================================


# ============================= Define the RNN model
input_size = 3  # Number of features
hidden_size = 25  # We can tune this
output_size = 3  # Predicting 3 features
rnn_model = RNNModel(input_size, hidden_size, output_size , s0_nature='zeros_', train_s0 = False, num_layers = 1)

unique_simulation_id = "s010"
num_epochs =500
scenario = RNNs_modelling_scenario(context_window_size=context_window_size, 
                                   forecast_horizon=forecast_horizon, num_epochs=num_epochs , simulation_id=unique_simulation_id)
# scenario.model_name(rnn_model)
learning_rate = 1e-3
optimizer = torch.optim.Adam(rnn_model.parameters() , lr = learning_rate)
criterion = torch.nn.MSELoss()
# rez_003 =  scenario.traintest_savebest( rnn_model ,train_loader , context_window_l, true_testdata_l, criterion, optimizer)
rez_004 =  scenario.train_only( rnn_model ,train_loader ,  criterion, optimizer)

# rnn_model.load_state_dict( rez_003['model_'])
# rnn_model.eval()
# pred_testdata__ = scenario.test_only(rnn_model, context_window_r , forecast_horizon )
# PlotTool.plotting_Double_ChaoticTrajectories(pred_testdata__.detach().numpy(), true_testdata_r.detach().numpy(), true_first=False)

# ========================================================================
# ========================================================================


input_size = 3  # Number of features
hidden_size = 40  # You can tune this
output_size = 3  # Predicting 3 features
lstm_model = LSTMModel(input_size, hidden_size, output_size , s0_nature='random_', train_s0 = True, num_layers = 1)

unique_simulation_id = "s010"
num_epochs =200
scenario = RNNs_modelling_scenario(context_window_size=context_window_size, 
                                   forecast_horizon=forecast_horizon, num_epochs=num_epochs , simulation_id=unique_simulation_id)

# scenario.model_name(lstm_model)
learning_rate = 1e-3
optimizer = torch.optim.Adam(lstm_model.parameters() , lr = learning_rate)
criterion = torch.nn.MSELoss()
# rez_004 =  scenario.traintest_savebest( lstm_model ,train_loader , context_window_l, true_testdata_l, criterion, optimizer)

rez_004 =  scenario.train_only( lstm_model ,train_loader ,  criterion, optimizer)

# lstm_model.load_state_dict( rez_004['model_'])
# lstm_model.eval()
# pred_testdata__ = scenario.test_only(lstm_model, context_window_r , forecast_horizon )
# PlotTool.plotting_Double_ChaoticTrajectories(pred_testdata__.detach().numpy(), true_testdata_r.detach().numpy() , true_first=False)