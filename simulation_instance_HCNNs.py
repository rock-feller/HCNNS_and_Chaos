import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from HCNNs_Climate.hcnn_modules.hcnn_models import Vanilla_ModelClimate , PTF_ModelClimate 
from HCNNs_Climate.hcnn_modules.hcnn_models import LSTM_FormulationClimate , LargeSparse_ModelClimate
from HCNNs_Climate.modeling_strategy.causal_modeling import HCNNs_Climate_modelling_scenario


from data.climate_wizard import processor , splitter , visual
climatix = processor.ClimateDataProcessor()
TransformAgent = splitter.SegmenterEngine()
PlotTool =  visual.ClimateDataVisalizer()


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ! Here we are going to use the winter data of 2021. You can uncomment [::6] to change the granularity of the data to 60min intervals
# ! Otherwise is 10min intervals

winter_df21 =  pd.read_csv('data/climate_data/season_n_sunrise_data/winter_full_2021.csv')[::6]
full_Tindex_and_data , avgs , stds  =  climatix.transform_and_assemble(winter_df21)
tens_winter21 =  torch.tensor(full_Tindex_and_data[:,1:].astype(np.float32)).unsqueeze(1)[-2000:]#[:1000]


####  ========= 10 min Interval ================
context_window =  144
forecast_horizon =  24
batch_size =  10
tensor_traindata = tens_winter21[:-forecast_horizon]
tensor_testdata = tens_winter21[-forecast_horizon:]
slided_traindataset =  splitter.SlidingWindowDataset(tensor_traindata , context_window)
train_loader =  splitter.SlidingWindowDataLoader( slided_traindataset , batch_size=batch_size , shuffle=False, device=device)



# last
ctxt_window_l , fcast_wdow_l, t_idx_ctxt_l , t_idx_fcast_l = TransformAgent.ctextwindow_testdata_generator( full_Tindex_and_data, context_window , forecast_horizon , 'last_')
#random
ctxt_window_r , fcast_wdow_r, t_idx_ctxt_r , t_idx_fcast_r = TransformAgent.ctextwindow_testdata_generator( full_Tindex_and_data[:-forecast_horizon], context_window , forecast_horizon , 'random_')
# df_or =  manip.map_final_output_to_original_dframe(ctxt_window[:,:4] , avgs, stds , t_idx_ctxt)


unique_id  =  "s001_Winter:_10min"
num_epochs =  100


# # ! ###### LSTM Formulation

# LSTMForm_net = LSTM_FormulationClimate(n_obs=4, n_hid_vars=25 , n_ext_vars=2,
#                                         s0_nature = "random_", train_s0=True)

# LSTMForm_net.to(device)

# scenario =HCNNs_Climate_modelling_scenario( context_window_size= context_window ,
#                                            forecast_horizon=forecast_horizon,
#                                            num_epochs= num_epochs,simulation_id= unique_id)


# learning_rate = 12e-3
# optimizer = torch.optim.Adam(LSTMForm_net.parameters(), lr=learning_rate)
# criterion  =  torch.nn.MSELoss()
# rez_lstm = scenario.abridged_mode(LSTMForm_net , train_loader= train_loader,context_window=ctxt_window_l, 
#                                  num_epochs= num_epochs , forecast_horizon= forecast_horizon,
#                                  true_testdata= fcast_wdow_l , criterion=criterion , optimizer=optimizer)

# ! ###### LSTM Formulation
print(" =============================ON TO THE NEXT ONE VanModel abridged=============================")

VanModel = Vanilla_ModelClimate(n_obs=4, n_hid_vars=100 , n_ext_vars=2,
                                        s0_nature = "random_", train_s0=True)

VanModel.to(device)

scenario =HCNNs_Climate_modelling_scenario( context_window_size= context_window ,
                                           forecast_horizon=forecast_horizon,
                                           num_epochs= num_epochs,simulation_id= unique_id)


learning_rate = 1e-3
optimizer = torch.optim.Adam(VanModel.parameters(), lr=learning_rate)
criterion  =  torch.nn.MSELoss()
rez_lstm = scenario.abridged_mode(VanModel , train_loader= train_loader,context_window=ctxt_window_l, 
                                 num_epochs= num_epochs , forecast_horizon= forecast_horizon,
                                 true_testdata= fcast_wdow_l , criterion=criterion , optimizer=optimizer)
print(" =============================ON TO THE NEXT ONE VanModel trainonly=============================")

VanModel = Vanilla_ModelClimate(n_obs=4, n_hid_vars=100 , n_ext_vars=2,
                                        s0_nature = "random_", train_s0=True)
VanModel = Vanilla_ModelClimate(n_obs=4, n_hid_vars=100 , n_ext_vars=2,
                                        s0_nature = "random_", train_s0=True)
rez_001 =  scenario.train_only(VanModel , train_loader= train_loader, num_epochs= num_epochs , criterion=criterion , optimizer=optimizer)








print(" =============================ON TO THE NEXT ONE Partial Teacher Forcing abridged=============================")


# ! ###### Partial Teacher Forcing


# PTF_Model01 = PTF_ModelClimate(n_obs=4, n_hid_vars=100 , n_ext_vars=2, 
#                                 s0_nature='random_',  train_s0=True , 
#                                 num_epochs=10 , target_prob=0.25 , drop_output=True)
# PTF_Model01.to(device)
# learning_rate = 1e-3
# optimizer = torch.optim.Adam(PTF_Model01.parameters(), lr=learning_rate)
# criterion  =  torch.nn.MSELoss()

# rez_002 = scenario.abridged_mode(PTF_Model01 , train_loader= train_loader,context_window=ctxt_window_r, 
#                                  num_epochs= num_epochs , forecast_horizon= forecast_horizon,
#                                  true_testdata= fcast_wdow_r , criterion=criterion , optimizer=optimizer)

print(" =============================ON TO THE NEXT ONE Partial Teacher Forcing train_only=============================")
# rez_004 =  scenario.train_only(PTF_Model01 , train_loader= train_loader, num_epochs= num_epochs , criterion=criterion , optimizer=optimizer)