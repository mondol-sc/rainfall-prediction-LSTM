# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""

import pandas as pd
import numpy as np
from my_custom_functions import write_description_vector_dataframe
from datetime import datetime

input_file_directory = "../Data/1. CleanTimeseriesData/"
export_file_directory = "../Data/2. ExtractedTabulatedData/"
program_name = 'code03_extraction_tabulation.py'

stations = ['sylhet', 'srimangal']
start_timeframe = {stations[0]: pd.Timestamp(year=1956, month=1, day=1),
                      stations[1]: pd.Timestamp(year=1950, month=1, day=1)}
end_timeframe = {stations[0]: pd.Timestamp(year=2022, month=1, day=1),
                      stations[1]: pd.Timestamp(year=2022, month=1, day=1)}
selected_hydromet_variables = ['total_rainfall', 'avg_cloud_amount',
                               'avg_dewpoint_temperature',  'avg_humidity',
                               'avg_msl_pressure', 'avg_sl_pressure',
                               'avg_temperature', 'maximum_temperature',
                               'minimum_temperature',
                               'prevailing_wind_speed_east_component',
                               'prevailing_wind_speed_north_component']
selected_teleconnection_params = ['teleconnection_DMI', 'teleconnection_Nino34',
                                  'teleconnection_PDO', 'teleconnection_SOI']

for station in stations:
    readme_file_name = export_file_directory + '/' + station +  "_readme.txt"
    with open(readme_file_name, 'w') as readme_file:
        datetime_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        readme_file.write("Created with the program: " + program_name + "\n" + 
                   "Date and Time: " + datetime_string + "\n" + 
                   "Author: Sujan Chandra Mondol\n")
        station_aggregated_dataframe = pd.DataFrame(index=pd.date_range(
            start=start_timeframe[station], end=end_timeframe[station], freq='MS'))
        start_timestamp = start_timeframe[station]
        end_timestamp = end_timeframe[station]
        for variable in selected_hydromet_variables:
            file_source = input_file_directory + station + '_' + variable + '.csv'
            file_data = pd.read_csv(file_source, header=None, index_col=0)
            file_data.index =[pd.Timestamp(idx) for idx in file_data.index]
            new_column = file_data[1][start_timestamp: end_timestamp]
            station_aggregated_dataframe[variable] = new_column
            heading = station + "_" + variable
            write_description_vector_dataframe(new_column, heading, readme_file)
        for param in selected_teleconnection_params:
            file_source = input_file_directory + param + '.csv'
            file_data = pd.read_csv(file_source, header=None, index_col=0)
            file_data.index =[pd.Timestamp(idx) for idx in file_data.index]
            new_column = file_data[1][start_timestamp: end_timestamp]
            station_aggregated_dataframe[param] = new_column
            heading = station + "_" + param
            write_description_vector_dataframe(new_column, heading, readme_file)
        export_file_name = station + "_" + 'extracted_tabulated_dataframe' + '.csv'
        station_aggregated_dataframe.to_csv(export_file_directory + export_file_name,
                                            header=True,
                                            index=True)
        print("The tabulated file namely [[" +export_file_name+ "]] has been"
              + " created successfully...")

                                    
        