
import torch
import time
import torch.nn as nn
import os
from datetime import date
import os, json , time
from copy import deepcopy
from typing import Optional, Type , Literal , List , Tuple , Dict
import pandas as pd
# def convert_sec(seconds):
#     minutes, seconds = divmod(seconds, 60)
#     hours, minutes = divmod(minutes, 60)

#     return f'{int(hours)}h {int(minutes)} min {round(seconds,2)}s'
    


class HCNNs_Climate_modelling_scenario():

    #folder creation for Vanilla HCNN models
    folder_pre_trained_van = os.makedirs("results_climate_hcnns/abridged_models/van/",  exist_ok=True)
    folder_pre_tested_van = os.makedirs("results_climate_hcnns/fully_unfolded_models/van/", exist_ok=True)
    folder_trained_based_van = os.makedirs("results_climate_hcnns/train_based_models/van/", exist_ok=True)
    #folder creation for HCNN-pTF models
    folder_pre_trained_ptf = os.makedirs("results_climate_hcnns/abridged_models/ptf/",  exist_ok=True)
    folder_pre_tested_ptf = os.makedirs("results_climate_hcnns/fully_unfolded_models/ptf/", exist_ok=True)
    folder_trained_based_ptf = os.makedirs("results_climate_hcnns/train_based_models/ptf/",  exist_ok=True)

    #folder creation for HCNN-Lform models
    folder_pre_trained_lstmform = os.makedirs("results_climate_hcnns/abridged_models/lstmform/",  exist_ok=True)
    folder_pre_tested_lstmform = os.makedirs("results_climate_hcnns/fully_unfolded_models/lstmform/", exist_ok=True)
    folder_trained_based_lstmform = os.makedirs("results_climate_hcnns/train_based_models/lstmform/",  exist_ok=True)

    #folder creation for HCNN-LSpa models
    folder_pre_trained_laSpa = os.makedirs("results_climate_hcnns/abridged_models/laSpa/",  exist_ok=True)
    folder_pre_tested_laSpa  = os.makedirs("results_climate_hcnns/fully_unfolded_models/laSpa/", exist_ok=True)
    folder_trained_based_laSpa = os.makedirs("results_climate_hcnns/train_based_models/laSpa/",  exist_ok=True)

    folder_output_dicts = os.makedirs("results_climate_hcnns/output_dicts/",  exist_ok=True)
    folder_output_csv= os.makedirs("results_climate_hcnns/csv_files/",  exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def __init__(self, context_window_size: int, forecast_horizon:int, num_epochs:int, simulation_id:str ):
                
            self.context_window_size = context_window_size
            self.forecast_horizon = forecast_horizon
            self.num_epochs = num_epochs
            self.simulation_id = simulation_id


    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def model_name( self , hcnn_instance  )-> str:

        """The model naming convention is 
        modeltype:_modelname_hd:_hidden_vars_ctxt:_ctxt_window_fc:_forecasthor_date_hour_
        """

        model_type = hcnn_instance.__class__.__name__
        hidden_vars = hcnn_instance.n_hid_vars
        #ctext_window =  self.context_window
        fcast_hor = self.forecast_horizon
        #today = date.today()
        # print(today.strftime("%d_%m"))
        if hcnn_instance.name == "ptf":
            model_full_name = f"{self.simulation_id}:_{model_type}:_hd{hidden_vars}{hcnn_instance.train_s0}:_ctxt{self.context_window_size}:_fc{fcast_hor}steps_TargetProb:_{hcnn_instance.target_prob}:_{self.num_epochs}epochs_trained_on_{date.today().strftime('%d_%m')}"
        elif hcnn_instance.name == "largesparse":
            
            model_full_name = f"{self.simulation_id}:_{model_type}:_hd{hidden_vars}{hcnn_instance.train_s0}:_ctxt{self.context_window_size}:_fc{fcast_hor}steps_TargetProb:_{hcnn_instance.pct_zeroed_weights}:_{self.num_epochs}epochs_trained_on_{date.today().strftime('%d_%m')}"
        else:
            model_full_name = f"{self.simulation_id}:_{model_type}:_hd{hidden_vars}{hcnn_instance.train_s0}:_ctxt{self.context_window_size}:_fc{fcast_hor}steps:_{self.num_epochs}epochs_trained_on_{date.today().strftime('%d_%m')}"

        #_at_{datetime.now().strftime("%H_%M_")}
        self.set_seed(42)
        return model_full_name

        # pd.DataFrame(results_dictionary['pred_testdata'], columns=['pred_wd20x', 'pred_wd20y', 'pred_ws60x','pred_wd60y' ]).to_csv(f"{output_csv}/{hcnn_instance.__class__.__name__}_{self.simulation_id}_pred.csv" , index=False)
        # pd.DataFrame(results_dictionary['true_testdata'], columns=['true_wd20x', 'true_wd20y', 'true_ws60x','true_wd60y']).to_csv(f"{output_csv}/{hcnn_instance.__class__.__name__}_{self.simulation_id}_true.csv" , index=False)
        # return print("Training results saved successfully")


    def results_saver(self, hcnn_instance, results_dictionary ):
        results_dictionary_cp = {k: v for k, v in results_dictionary.items() if k not in  ["model_", "true_testdata", "pred_testdata","s_states" , "diag_params"]}

        output_dicts = "results_climate_hcnns/output_dicts/"
        output_csv = "results_climate_hcnns/csv_files/"

        with open(f"{output_dicts}{hcnn_instance.__class__.__name__}_{self.simulation_id}.json", "w") as file:
            json.dump(results_dictionary_cp, file)

        pd.DataFrame(results_dictionary['pred_testdata'], columns=['pred_wd20x', 'pred_wd20y', 'pred_ws60x','pred_wd60y' ]).to_csv(f"{output_csv}/{hcnn_instance.__class__.__name__}_{self.simulation_id}_pred.csv" , index=False)
        pd.DataFrame(results_dictionary['true_testdata'], columns=['true_wd20x', 'true_wd20y', 'true_ws60x','true_wd60y']).to_csv(f"{output_csv}/{hcnn_instance.__class__.__name__}_{self.simulation_id}_true.csv" , index=False)
        pd.DataFrame(results_dictionary['s_states'], columns=[f"hid_state_{i+1}" for i in range(hcnn_instance.n_hid_vars)]).to_csv(f"{output_csv}/{hcnn_instance.__class__.__name__}_{self.simulation_id}_states.csv" , index=False)
        if hcnn_instance.name == "lstm_form":
            pd.DataFrame(results_dictionary['diag_params'], columns=["diags_params"]).to_csv(f"{output_csv}/{hcnn_instance.__class__.__name__}_{self.simulation_id}_diags.csv" , index=False)
        return print("Training results saved successfully || results dicts, json's and csv's created successfully: check the results folder")
    
    def target_externals_separator( self,  true_testdata : torch.Tensor , n_outputs) -> Tuple[torch.Tensor]:
        if true_testdata.dim() == 3:
            actual_true_test_data = true_testdata[:, :, :n_outputs]#.squeeze(2)
            actual_true_externals = true_testdata[:,:, n_outputs:]#.squeeze(2).unsqueeze(1).unsqueeze(1)
        elif true_testdata.dim() ==2:
            true_testdata = true_testdata.unsqueeze(1)
            actual_true_test_data = true_testdata[:, :,:n_outputs]#.squeeze(2)
            actual_true_externals = true_testdata[:, :, n_outputs:]#.squeeze(2).unsqueeze(1).unsqueeze(1)

        return actual_true_test_data , actual_true_externals

    def test_only( self, hcnn_instance:nn.Module , context_window:torch.Tensor, 
                  forecast_horizon :int ,  tensor_externals: torch.Tensor) -> Tuple[torch.Tensor]:

        self.set_seed(42)
        hcnn_instance.eval()

            #hook_handle.remove()

        #with torch.no_grad():


            #out_clust, r_states, s_states = model.forward( train_data )
        initial_state = hcnn_instance.initial_hidden_state()
        preds , fs_states = hcnn_instance.forecast( initial_state, context_window , forecast_horizon , tensor_externals)
        
        return preds , fs_states

    def fully_unfolded(self, hcnn_instance:nn.Module, train_data : torch.Tensor ,true_testdata : torch.Tensor , 
                       criterion :nn.Module , optimizer :nn.Module , current_epoch:Optional[int]=None, prob: Optional[int]=None) -> Dict:

        self.set_seed(42)
        folder_fullyunf = "results_climate_hcnns/fully_unfolded_models/"
        #device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # context_window =  all_data[:contextsize_for_inference].to(self.device)

        # test_data = all_data[contextsize_for_inference: contextsize_for_inference + forecast_horizon].to(self.device)

        best_val_loss = float('inf')
        train_start_time = time.time()
        # training_losses = []
        results_dict = {}

        prob = 0.
        #avgs_training_losses_perepoch = []
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()


            hcnn_instance.train()  # Set the model to training mode

            #hook_handle = model.register_forward_hook(cut_off_future_tsteps_hook)

            

            #for observed_data in  train_data:

            optimizer.zero_grad()
            out_target = torch.zeros_like(train_data)
            #out_target = train_data.clone()


            if hcnn_instance.name == "ptf":
                initial_state_vector = hcnn_instance.initial_hidden_state(batch_size=1)
                prob = hcnn_instance.decrease_dropout_prob(epoch, prob)
                Xions , out_clust,  s_states = hcnn_instance.forward( initial_state_vector, train_data ,prob)
            else:
                initial_state_vector = hcnn_instance.initial_hidden_state(batch_size=1)
                Xions , out_clust,  s_states = hcnn_instance.forward(initial_state_vector,  train_data)
            # __ , out_clust,  s_states = hcnn_instance.forward( train_data)


            
            training_loss  = criterion(out_clust, out_target)
            #training_loss  = criterion(out_clust, train_data)
            print (f"Epoch {epoch+1} || train_loss= {round(training_loss.item(),4)}")
            training_loss.backward()


            optimizer.step()

            #training_losses.append(training_loss.detach().item())
            

            epoch_end_time = time.time()

            epoch_duration = epoch_end_time - epoch_start_time

            
            # print(f"Epoch {epoch + 1}/{num_epochs} completed in {convert_sec(epoch_elapsed_time)}")
            #print(p1_)

            pred_testdata , fs_states = self.test_only( hcnn_instance, train_data , forecast_horizon)
            # hcnn_instance.eval()
            # #hook_handle.remove()
            # with torch.no_grad():
            #     #out_clust, r_states, s_states = model.forward( train_data )
            #     f_Xions , fs_states = hcnn_instance.forecast( context_window , forecast_horizon)
            #     #print( "at inference , prob  = " , hcnn_instance.prob)

            test_loss = criterion(pred_testdata[-forecast_horizon:], true_testdata).item()#test_data
            results_dict[f'epoch_{epoch +1}'] = {'training_loss': training_loss.item(), 'test_loss': test_loss,
                                                 'time_taken': epoch_duration}
            
            

            if test_loss < best_val_loss:
                results_dict.update({
                    'best_epoch': epoch + 1,
                    "best_Te_loss": test_loss,
                    "model_": hcnn_instance.state_dict(),
                    'pred_testdata': pred_testdata.squeeze(1).detach().cpu().numpy(),

                    # 'model_': torch.save(hcnn_instance , "hcnn_ful_unf_trained_on_"+dd_mm+".pth"),
                    's_states': fs_states.squeeze(1).detach().cpu().numpy()

                    #'f_states': fr_states,
                                })

                best_val_loss = test_loss

            if (epoch+1) % 1 == 0:
                print(f'Epoch {epoch +1}/{self.num_epochs} completed || Time taken: {round(epoch_duration,2)} seconds || best epoch : {results_dict["best_epoch"]}')
                print(f'current test loss : {round(test_loss, 4)}|| best test loss : {round(results_dict["best_Te_loss"], 4)}')

            else:
                pass

            # if hcnn_instance.name == "ptf":
                
            #     hcnn_instance.decrease_dropout_prob(epoch , self.num_epochs , 0.25)
            # else:
            #     pass
            
            epoch_elapsed_time = time.time() - epoch_start_time
            # print(f"Best val_loss = {round(results['te_loss'], 4)} is at Epoch {results_dict['epoch']} ||  curr_val_loss = {round(val_loss,4)}")
            # print('\n')
            # print (" =========================== ")

        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)
        print(f'Training completed. Total time taken: {total_time} seconds')
        results_dict['Total_TrainingTime'] = total_time
        # print(f"total_train_time = {convert_sec( total_train_time)}")
        results_dict['true_testdata'] = true_testdata.squeeze(1).detach().cpu().numpy()
        results_dict["model_name:"]= self.model_name(hcnn_instance)
        torch.save(deepcopy(results_dict["model_"]) , f"{folder_fullyunf}Te_{self.model_name(hcnn_instance)}.pt")
        
        self.results_saver(hcnn_instance , results_dict)
        return  results_dict



    def abridged_mode(self, hcnn_instance: nn.Module,  train_loader ,context_window:torch.Tensor, num_epochs:int, 
                            true_testdata:torch.Tensor, forecast_horizon:int, criterion :nn.Module , optimizer :nn.Module )->  Dict:

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # context_window =  all_data[:contextsize_for_inference].to(device)

        # test_data = all_data[contextsize_for_inference: contextsize_for_inference + forecast_horizon].to(device)

        """ The train function """
        best_test_loss = float('inf')
        results_dict={}
        train_start_time = time.time()
        training_losses = []
        self.set_seed(42)
        folder_pre_trained_van = "results_climate_hcnns/abridged_models/van/"
        folder_pre_trained_ptf = "results_climate_hcnns/abridged_models/ptf/"
        folder_pre_trained_lstmform = "results_climate_hcnns/abridged_models/lstmform/"
        folder_pre_trained_laSpa = "results_climate_hcnns/abridged_models/laSpa/"
        prob  = 0.

        actual_true_testdata , tens_future_externals = self.target_externals_separator (true_testdata , hcnn_instance.n_obs)
        for epoch in range(num_epochs):

            epoch_start_time = time.time()


            for  batch_id, batch_data in enumerate( train_loader):
                
                hcnn_instance.train()  # Set the model to training mode

                optimizer.zero_grad()

                out_target = torch.zeros(batch_data.size(0),batch_data.size(1), 1, hcnn_instance.n_obs , dtype=torch.float32 , device=self.device)


                # _x_, out_clust,  s_states = hcnn_instance.forward( batch_data )
                
                if hcnn_instance.name == "ptf":
                    initial_state_vector = hcnn_instance.initial_hidden_state(batch_size=batch_data.shape[0])
                    prob = hcnn_instance.decrease_dropout_prob(epoch, prob)
                    Xions , out_clust,  s_states = hcnn_instance.forward( initial_state_vector, batch_data ,prob )
                
                else:
                    initial_state_vector = hcnn_instance.initial_hidden_state(batch_size=batch_data.shape[0])
                    Xions , out_clust,  s_states = hcnn_instance.forward(initial_state_vector,  batch_data )

                #print(out_clust)

                # training_loss  = criterion(out_clust, out_target)
                batch_loss  = criterion(out_clust, out_target)

                batch_loss.backward()


                optimizer.step()
                training_losses.append(batch_loss.item())

                hcnn_instance.eval()

                pred_testdata , fs_states = self.test_only( hcnn_instance, context_window.unsqueeze(1) , forecast_horizon , tens_future_externals)

                test_loss = criterion(pred_testdata[-forecast_horizon:], actual_true_testdata).item()#test_data

                if test_loss < best_test_loss:
                    results_dict.update({'best_epoch':epoch +1 , "best_Te_loss": test_loss, 
                                        "model_": hcnn_instance.state_dict(),
                                        "pred_testdata": pred_testdata[-forecast_horizon:].squeeze(1).detach().cpu().numpy(),
                                        's_states': fs_states.squeeze(1).detach().cpu().numpy()})
                    
                    if hcnn_instance.name =="lstm_form":
                        results_dict.update({"diag_params": hcnn_instance.state_dict()['cell.D.weight'].diag().detach().cpu().numpy()})

                    best_test_loss = test_loss

            avg_training_loss = sum(training_losses)/len(training_losses)
            epoch_end_time = time.time()
            epoch_duration = round(epoch_end_time - epoch_start_time , 2)
            results_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'time_taken': epoch_duration}

            if (epoch+1) % 1 == 0:
                print(f'Epoch {epoch +1}/{self.num_epochs} completed || Time taken: {epoch_duration} seconds || best epoch : {results_dict["best_epoch"]}')
                print(f'current test loss : {round(test_loss, 4)}|| best test loss : {round(results_dict["best_Te_loss"], 4)}')

            else:
                pass
        
        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)
        print(f'Training completed. Total time taken: {total_time} seconds')
        results_dict['Total_TrainingTime'] = total_time
        # print(f"total_train_time = {convert_sec( total_train_time)}")
        results_dict['true_testdata'] = true_testdata.squeeze(1).detach().cpu().numpy()[:, :hcnn_instance.n_obs]
        results_dict["model_name:"]= self.model_name(hcnn_instance)
        if hcnn_instance.name == "ptf":
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_pre_trained_ptf}Te_{self.model_name(hcnn_instance)}.pt")
        elif hcnn_instance.name == "lstm_form":
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_pre_trained_lstmform}Te_{self.model_name(hcnn_instance)}.pt")
        elif hcnn_instance.name == "largesparse":
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_pre_trained_laSpa}Te_{self.model_name(hcnn_instance)}.pt")
        else:
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_pre_trained_van}Te_{self.model_name(hcnn_instance)}.pt")
        
        self.results_saver(hcnn_instance , results_dict)
        return  results_dict
    

    def train_only(self, hcnn_instance: nn.Module , train_loader , num_epochs:int , optimizer: nn.Module, criterion: nn.Module):

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # context_window =  all_data[:contextsize_for_inference].to(device)

        # test_data = all_data[contextsize_for_inference: contextsize_for_inference + forecast_horizon].to(device)

        """ The train function """
        best_tr_loss = float('inf')
        results_dict={}
        train_start_time = time.time()
        training_losses = []
        self.set_seed(42)
        folder_trained_based_van = "results_climate_hcnns/train_based_models/van/"
        folder_trained_based_ptf = "results_climate_hcnns/train_based_models/ptf/"
        folder_trained_based_lstmform = "results_climate_hcnns/train_based_models/lstmform/"
        folder_trained_based_laSpa = "results_climate_hcnns/train_based_models/laSpa/"
        prob  = 0.

        # actual_true_testdata , tens_future_externals = self.target_externals_separator (true_testdata , hcnn_instance.n_obs)
        for epoch in range(num_epochs):

            epoch_start_time = time.time()


            for  batch_id, batch_data in enumerate( train_loader):
                
                hcnn_instance.train()  # Set the model to training mode

                optimizer.zero_grad()

                out_target = torch.zeros(batch_data.size(0),batch_data.size(1), 1, hcnn_instance.n_obs , dtype=torch.float32 , device=self.device)


                # _x_, out_clust,  s_states = hcnn_instance.forward( batch_data )
                
                if hcnn_instance.name == "ptf":
                    initial_state_vector = hcnn_instance.initial_hidden_state(batch_size=batch_data.shape[0])
                    prob = hcnn_instance.decrease_dropout_prob(epoch, prob)
                    Xions , out_clust,  s_states = hcnn_instance.forward( initial_state_vector, batch_data ,prob )
                
                else:
                    initial_state_vector = hcnn_instance.initial_hidden_state(batch_size=batch_data.shape[0])
                    Xions , out_clust,  s_states = hcnn_instance.forward(initial_state_vector,  batch_data )

                #print(out_clust)

                # training_loss  = criterion(out_clust, out_target)
                batch_loss  = criterion(out_clust, out_target)

                batch_loss.backward()


                optimizer.step()
                training_losses.append(batch_loss.item())

                # hcnn_instance.eval()

                # pred_testdata , fs_states = self.test_only( hcnn_instance, context_window.unsqueeze(1) , forecast_horizon , tens_future_externals)
                #  training_losses.append(train_loss.item())
                avg_training_loss = sum(training_losses)/len(training_losses)
                epoch_end_time = time.time()
                epoch_duration = round(epoch_end_time - epoch_start_time , 2)

                results_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'time_taken': epoch_duration}

                if avg_training_loss< best_tr_loss:
                    # torch.save(custom_model.state_dict() , f"{folder_pre_trained}Tr_{self.model_name(custom_model)}.pth")
                    results_dict.update({'best_Tr_model at':epoch +1 , "best_AvgTr_loss": avg_training_loss,
                                        "model_": hcnn_instance.state_dict()})
                    best_tr_loss = avg_training_loss


                    
            if (epoch+1) % 1 == 0:

                # print(f'Epoch {epoch +1} completed. Time taken: {epoch_time} seconds')
                print(f'Epoch {epoch +1} completed || Time taken: {epoch_duration} seconds || best epoch : {results_dict["best_Tr_model at"]}')
                print(f'current training loss : {round(avg_training_loss, 4)}|| best training loss : {round(results_dict["best_AvgTr_loss"], 4)}')

                results_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'time_taken': epoch_duration}

                


        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)

        if hcnn_instance.name == "ptf":
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_trained_based_ptf}Tr_{self.model_name(hcnn_instance)}.pt")
        elif hcnn_instance.name == "lstm_form":
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_trained_based_lstmform}Tr_{self.model_name(hcnn_instance)}.pt")
        elif hcnn_instance.name == "largesparse":
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_trained_based_laSpa}Tr_{self.model_name(hcnn_instance)}.pt")
        else:
            torch.save(deepcopy(results_dict["model_"]) , f"{folder_trained_based_van}Tr_{self.model_name(hcnn_instance)}.pt")


        print(f'Training completed. Total time taken: {total_time} seconds')
        results_dict['Total_TrainingTime'] = total_time

        return results_dict
        
    
# LSTMForm_net.state_dict()['cell.D.weight'].diag().numpy().tolist()
