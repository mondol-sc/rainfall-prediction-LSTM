# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""

import pandas as pd
import numpy as np
from my_custom_functions import getting_files_of_selected_extensions

input_file_directory = "../Data/1. CleanTimeseriesData/"
all_files_list = getting_files_of_selected_extensions(input_file_directory, ".csv")
stations = ['sylhet', 'srimangal']

direction_angle_dict = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, 
                        "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180, 
                        "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, 
                        "WNW": 292.5, "NW": 315, "NNW": 337.5}

for file_name in all_files_list:
    for station in stations:
        if (station + "_prevailing_wind_direction") in file_name:
            speed_file_directory = input_file_directory + (
                station + "_prevailing_wind_speed") + ".csv"
            speed_dataframe = pd.read_csv(speed_file_directory, 
                                          header=None, index_col=0)
            direction_file_directory = input_file_directory + (
                station + "_prevailing_wind_direction") + ".csv"
            direction_dataframe = pd.read_csv(direction_file_directory, 
                                              header=None, index_col=0)
            
            # Initiate Export DataFrames
            north_components_df = speed_dataframe.copy()
            east_components_df = speed_dataframe.copy()
            
            for idx in speed_dataframe.index:
                if direction_dataframe[1][idx] == "CLM":
                    north_components_df[1][idx] = 0
                    east_components_df[1][idx] = 0
                elif pd.isna(direction_dataframe[1][idx]) == True or (
                        pd.isna(speed_dataframe[1][idx]) == True):
                    north_components_df[1][idx] = np.nan
                    east_components_df[1][idx] = np.nan
                else:
                    angle = direction_angle_dict[(direction_dataframe[1][idx])
                                                  .strip()]
                    north_components_df[1][idx] = speed_dataframe[1][idx] * (
                        np.cos(np.pi*angle/180))
                    east_components_df[1][idx] = speed_dataframe[1][idx] * (
                        np.sin(np.pi*angle/180))
            
            north_components_df.to_csv(input_file_directory + "/" + station + "_" +
                                   "prevailing_wind_speed_north_component.csv",
                                   header=None)
            east_components_df.to_csv(input_file_directory + "/" + station + "_" +
                                   "prevailing_wind_speed_east_component.csv",
                                   header=None)
            print("Component files for [[" + station  + "]] has been successfully created...")

    
    