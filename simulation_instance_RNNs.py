import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from RNNs_Climate.models.rnn_based_models import RNNModelClimate , LSTMModelClimate
from RNNs_Climate.modeling_strategy.causal_modeling import RNNs_ClimateModelling_scenario


from HCNNs_Climate.hcnn_modules.hcnn_models import LSTM_FormulationClimate , LargeSparse_ModelClimate
from HCNNs_Climate.modeling_strategy.causal_modeling import HCNNs_Climate_modelling_scenario


from data.climate_wizard import processor , splitter , visual
climatix = processor.ClimateDataProcessor()
TransformAgent = splitter.SegmenterEngine()
PlotTool =  visual.ClimateDataVisalizer()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ! Here we are going to use the winter data of 2021. You can uncomment [::3] to change the granularity of the data to 30min intervals
# ! Otherwise is 10min intervals
winter_df21 =  pd.read_csv('data/climate_data/season_n_sunrise_data/winter_full_2021.csv')#[::3]
full_Tindex_and_data , avgs , stds  =  climatix.transform_and_assemble(winter_df21)
tens_winter21 =  torch.tensor(full_Tindex_and_data[:,1:].astype(np.float32)).unsqueeze(2)[:500]
# tens_winter21.shape 


####  ========= 10 min Interval ================
context_window_size =  144
forecast_horizon =  24
batch_size =  10 
tensor_traindata = tens_winter21[:].squeeze(2) #forecast_horizon
tensor_testdata = tens_winter21[-forecast_horizon:].squeeze(2)

train_loader = splitter.InpTarg_TSDataset.create_train_loader(tensor_traindata, context_window_size, batch_size, shuffle=False)



# last
ctxt_window_l , fcast_wdow_l, t_idx_ctxt_l , t_idx_fcast_l = TransformAgent.ctextwindow_testdata_generator( full_Tindex_and_data, context_window_size , forecast_horizon , 'last_')
#random
ctxt_window_r , fcast_wdow_r, t_idx_ctxt_r , t_idx_fcast_r = TransformAgent.ctextwindow_testdata_generator( full_Tindex_and_data[:-forecast_horizon], context_window_size , forecast_horizon , 'random_')
# df_or =  manip.map_final_output_to_original_dframe(ctxt_window[:,:4] , avgs, stds , t_idx_ctxt)


# ============================= Define the RNN model
input_size = 6  # Number of features
hidden_size = 100  # We can tune this
output_size = 4  # Predicting 3 features

unique_id  =  "s001_Winter2021:_10min"
num_epochs =100


# !  # ============================= Define the LSTM model 
lstm_model = LSTMModelClimate(input_size, hidden_size, output_size , s0_nature='random_', train_s0 = False, num_layers = 1)


scenario = RNNs_ClimateModelling_scenario(context_window_size=context_window_size, 
                                   forecast_horizon=forecast_horizon, num_epochs=num_epochs , simulation_id=unique_id)
# scenario.model_name(rnn_model)
learning_rate = 1e-3
optimizer = torch.optim.Adam(lstm_model.parameters() , lr = learning_rate)
criterion = torch.nn.MSELoss()
rez_001te =  scenario.traintest_savebest( lstm_model ,train_loader , ctxt_window_l , fcast_wdow_l, criterion, optimizer) 
print(" =============================ON TO THE NEXT ONE =============================")

# rez_001tr = scenario.train_only( lstm_model ,train_loader,criterion, optimizer , num_epochs)




# ! ============================= Define the RNN model
input_size = 6  # Number of features
hidden_size = 100  # We can tune this
output_size = 4  # Predicting 3 features

rnn_model = RNNModelClimate(input_size, hidden_size, output_size , s0_nature='random_', train_s0 = False, num_layers = 1)


scenario = RNNs_ClimateModelling_scenario(context_window_size=context_window_size, 
                                   forecast_horizon=forecast_horizon, num_epochs=num_epochs , simulation_id=unique_id)
# scenario.model_name(rnn_model)
learning_rate = 1e-3
optimizer = torch.optim.Adam(rnn_model.parameters() , lr = learning_rate)
criterion = torch.nn.MSELoss()
rez_002te =  scenario.traintest_savebest( rnn_model ,train_loader , ctxt_window_l , fcast_wdow_l, criterion, optimizer) 
print(" =============================ON TO THE NEXT ONE =============================")
# rez_002tr = scenario.train_only( rnn_model ,train_loader,criterion, optimizer , num_epochs)