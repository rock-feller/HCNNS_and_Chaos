import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class sunshine_data_parser():
    def __init__(self ):
        #self.file_path =  file_path
        self.data_in_folder = True


    def tidy_Sunhours_data(self , file_path ) -> pd.DataFrame:

        data_frame = pd.read_csv(file_path)
        month_year = file_path.split("/")[-1].split(".")[0]
        month_id =  file_path.split("/")[-1].split("_")[1][:2]

        data_frame['Date'] = pd.date_range(data_frame.columns[0]+ "20"+
                                        month_id, periods=data_frame.shape[0])
        
        data_frame = data_frame.iloc[:, [1, 2 , 3, -1]]
        data_frame = data_frame[[data_frame.columns[-1]] + list(data_frame.columns[0:3])]
    
        data_frame['rise_adj'] = pd.to_datetime(data_frame['Sunrise'].str[:5].apply(self.round_to_next_10th_minutes).str[:]+ ":00").dt.time
        data_frame['set_adj'] = pd.to_datetime(data_frame['Sunset'].str[:5].apply(self.round_to_next_10th_minutes).str[:]+ ":00").dt.time

        data_frame['Sunrise'] = pd.to_timedelta(data_frame['Sunrise'].str[:5]+":00")
        data_frame['Sunset'] = pd.to_timedelta(data_frame['Sunset'].str[:5]+":00")
        
        data_frame["calc_length"] = data_frame['Sunset']-data_frame['Sunrise']
        data_frame['calc_length'] = data_frame['calc_length'].apply(self.convert_timedelta_to_hours)
        
        data_frame["adj_length"] = pd.to_timedelta(data_frame['set_adj'].astype(str))- pd.to_timedelta(data_frame['rise_adj'].astype(str))
        data_frame['adj_length'] = data_frame['adj_length'].apply(self.convert_timedelta_to_hours)
        return  data_frame , month_year
    
    def convert_timedelta_to_hours(self, timedelta):
        """Converts a timedelta to a number of hours.

        Args:
            timedelta: A timedelta object.

        Returns:
            The number of hours in the timedelta.
        """

        total_seconds = timedelta.total_seconds()
        hours = total_seconds / 3600.0

        return hours

    def round_to_next_10th_minutes(self, time_string):
        """Rounds a time string to the next 10th minutes.

        Args:
        time_string: A time string in the format "HH:MM".

        Returns:
        A time string rounded to the next 10th minutes.
        """

        hours, minutes = time_string.split(":")
        minutes = int(minutes)

        # Round the minutes up to the next 10th minute.
        minutes_dec = (minutes // 10) * 10 #+ 10
        rem_ = minutes%10

        # If the minutes are greater than 59, then increment the hours and set the minutes to 0.
        if minutes > 55:
            hours = int(hours) + 1
            minutes =  0 #minutes_dec +10
            
        elif rem_ >5 :
            hours = int(hours)
            minutes = minutes_dec + 10
        
        else:
            hours = int(hours)
            minutes = minutes_dec
            

        # Return the rounded time string.
        return f"{hours:02d}:{minutes:02d}"
    
    def generate_date_range(self, date_str):
        # Parse the input string
        month, year = date_str.split('_')
        month = int(month)
        year = int('20' + year)  # Assuming the format is MM_YY for years 2000-2099
        
        # Generate the start and end dates
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = (start_date + pd.offsets.MonthEnd(0)) + pd.DateOffset(days=1) - pd.DateOffset(seconds=1)
        
        # Create the date range with 10-minute frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq='10min')
        
        # Create a DataFrame with the date range
        df = pd.DataFrame(date_range, columns=['Date'])
        
        return df

    def monthly_length_of_day_generator(self , file_path) -> pd.DataFrame:

        data_frame , month_year =  self.tidy_Sunhours_data(file_path)
        restored_date_column = self.generate_date_range(month_year)
        t_grid ="10min"
        n= int((60/int(t_grid[:2]))*24 -1)
        list_alldf = []
        for ind_ in range(data_frame.shape[0]):
            df_within = pd.DataFrame()
            df_within['time'] = pd.date_range(str(data_frame['rise_adj'].values[ind_]) , str(data_frame['set_adj'].values[ind_]), freq='10min' )
            df_within['time'] =  df_within['time'].dt.time
            df_within['t_index'] = pd.Series(np.tile(np.arange(1, n + 1), len(df_within)//n + 1)[:len(df_within)]).values

            df_bef =pd.DataFrame()
            df_bef['time'] = pd.date_range("00:00:00" , str(df_within.iloc[0,0]) , inclusive='left' , freq="10min")
            df_bef['time'] = df_bef['time'].dt.time
            df_bef['t_index'] =  np.zeros(df_bef.shape[0])
            df_aft =pd.DataFrame()
            df_aft['time'] = pd.date_range(str(df_within.iloc[-1,0]) , "23:50:00" , inclusive='right' , freq="10min")
            df_aft['time'] = df_aft['time'].dt.time
            df_aft['t_index'] =  np.zeros(df_aft.shape[0])
            list_alldf.append(df_bef)
            list_alldf.append(df_within)
            list_alldf.append(df_aft)
    
        df_total= pd.concat(list_alldf, axis = 0)
        df_total['time_ofDay_cos'] = np.cos(2*np.deg2rad(180)* df_total['t_index']/(24*int((60/int(t_grid[:2])))))
        df_total['time_ofDay_sin'] = np.sin(2*np.deg2rad(180)* df_total['t_index']/(24*int((60/int(t_grid[:2])))))
        df_total = df_total.reset_index(drop=True)
        df_total['time'] = restored_date_column.iloc[:,0]
        df_total = df_total.rename(columns={'time':'date_time'})
        return df_total 
    
    def yearly_length_of_day_generator( self , folder_path):
        #os.makedirs("./../"+ folder_path "results/pretrained_models/",  exist_ok=True)
        df_list = []
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = folder_path + file
                df = self.monthly_length_of_day_generator(file_path)
                df= df.reset_index(drop=True)
                df_list.append(df)
        df_total = pd.concat(df_list, axis = 0)
        df_total =  df_total.sort_values(by='date_time')
        df_total.to_csv('df_all_sunshours_' + folder_path.split("/")[-2] + '.csv', index=False)
        print(f" Done with the sunshine data for the {folder_path.split('/')[-2]}")
        return df_total



# tidier =  tidy_sunshiner()

# df_tidy =  tidier.monthly_length_of_day_generator(file_path="/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/raw_data_indexed_months/year_21/09_21.csv")

# df_full22 =  tidier.yearly_length_of_day_generator(folder_path ="/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/raw_data_indexed_months/year_22/")

# df_full21 =  tidier.yearly_length_of_day_generator(folder_path ="/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/raw_data_indexed_months/year_21/")

# df_full20 =  tidier.yearly_length_of_day_generator(folder_path ="/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/raw_data_indexed_months/year_20/")





# dv_ops  =  data_frame_ops()
# file_path_wind_20 = "/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/climate_data/clean_data/df_all_dec2020_10min.csv" 
# file_path_Tindex_20 = "/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/df_all_sunshours_year_20.csv"

# file_path_wind_21 = "/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/climate_data/clean_data/df_all2021_10min.csv" 
# file_path_Tindex_21 = "/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/df_all_sunshours_year_21.csv"

# file_path_wind_22 = "/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/climate_data/clean_data/df_all2022_10min.csv" 
# file_path_Tindex_22 = "/home/rockefeller/Documents/PhDlife/DS_ResearchGroup/Year24/07_24/data/sunhours_data/df_all_sunshours_year_22.csv"


# df_new_20 =  extract_merge_fcastvar_only(file_path_wind_20, file_path_Tindex_20)
# df_new_21 =  extract_merge_fcastvar_only(file_path_wind_21, file_path_Tindex_21)
# df_new_22 =  extract_merge_fcastvar_only(file_path_wind_22, file_path_Tindex_22)

# df_new_22['date_time']  = pd.to_datetime(df_new_22['date_time'] )- pd.Timedelta(hours=0, minutes=10, seconds=0)
# df_20_to_22 =  pd.concat([df_new_20, df_new_21, df_new_22], axis=0)
# df_20_to_22 =  df_20_to_22.reset_index(drop=True)
# df_20_to_22.head()
