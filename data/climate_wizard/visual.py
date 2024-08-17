import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from typing import Tuple , Literal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class ClimateDataVisalizer():

    def __init__(self):
        self.plotting = True

    
    def plotting_Double_WSWD_Trajectories( self,  df_true:pd.DataFrame , df_pred:pd.DataFrame , time_interval:int ,  true_first:bool = True):
        
        if true_first:

            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid of subplots

            # Plot configurations
            plots = [
                ('WS20M', 'WS20M', 'Wind Speed 20M'),  # (df_true column, df_pred column, title)
                ('WD20M', 'WD20M', 'Wind Direction 20M'),
                ('WS60M', 'WS60M', 'Wind Speed 60M'),
                ('WD60M', 'WD60M', 'Wind Direction 60M')
            ]



            for ax, (true_col, pred_col, title) in zip(axs.flat, plots):
                ax.plot(df_pred['date_time'], df_pred[pred_col], marker='o', label='Pred ' + pred_col)
                ax.plot(df_true['date_time'], df_true[true_col], marker='o', label='True ' + true_col)
                

                ax.xaxis.set_major_locator(ticker.MultipleLocator(time_interval))
                ax.grid(True)
                ax.legend()
                ax.set_xlabel('Time')
                ax.set_ylabel(true_col)
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)

            # Adjust layout to prevent overlap
            fig.tight_layout()
            plt.show()

        else:
            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid of subplots

            # Plot configurations
            plots = [
                ('WS20M', 'WS20M', 'Wind Speed 20M'),  # (df_true column, df_pred column, title)
                ('WD20M', 'WD20M', 'Wind Direction 20M'),
                ('WS60M', 'WS60M', 'Wind Speed 60M'),
                ('WD60M', 'WD60M', 'Wind Direction 60M')
            ]



            for ax, (true_col, pred_col, title) in zip(axs.flat, plots):
                ax.plot(df_true['date_time'], df_true[true_col], marker='o', label='True ' + true_col)
                ax.plot(df_pred['date_time'], df_pred[pred_col], marker='o', label='Pred ' + pred_col)

                

                ax.xaxis.set_major_locator(ticker.MultipleLocator(time_interval))
                ax.grid(True)
                ax.legend()
                ax.set_xlabel('Time')
                ax.set_ylabel(true_col)
                ax.set_title(title)
                ax.tick_params(axis='x', rotation=45)

            # Adjust layout to prevent overlap
            fig.tight_layout()
            plt.show()

            return


