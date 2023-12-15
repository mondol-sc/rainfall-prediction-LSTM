# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from my_custom_functions import write_description_vector_dataframe
from datetime import datetime

input_file_directory = "../Data/4. FeatureSelectedDataWithDescriptions/"
export_file_directory = "../Data/5. SplitScaledData/"
program_name = 'code06_splitting_scaling_data.py'

stations = ['sylhet', 'srimangal']

for station in stations:
    file_name = station + '_reduced_dataframe.csv' 
    tabular_dataframe = pd.read_csv(input_file_directory + file_name, 
                                    header=0, index_col=0) 
    tabular_dataframe.index = [pd.Timestamp(idx) for idx in tabular_dataframe.index]
    X_all = tabular_dataframe.iloc[:, :-1]
    y_all = tabular_dataframe.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                        test_size=0.20, 
                                                        shuffle=False)
    
    readme_file_name = export_file_directory + station + "_splitting_readme.txt"
    with open(readme_file_name, 'w') as readme_file:
        datetime_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        readme_file.write("Created with the program: " + program_name + "\n" + 
                   "Date and Time: " + datetime_string + "\n" + 
                   "Author: Sujan Chandra Mondol\n\n" + "TEST SIZE = 20% \n\n"
                   "X_train\n" + "=================================\n" + 
                   'Number of (Records, Variables)' + str(X_train.shape)+ '\n'+
                   str(X_train.columns) + "\n" + 'Start ' + 
                   str(X_train.index[0]) + '\n' + 'End ' + str(X_train.index[-1])
                   + "\n\n y_train\n" + "=================================\n" + 
                   'Number of (Records, Variables)' + str(y_train.shape)+ '\n'
                   'Start ' + str(y_train.index[0]) + '\n' + 'End ' + 
                   str(y_train.index[-1]) + '\n\n' + "X_test\n" + 
                   "=================================\n" + 
                   'Number of (Records, Variables)' + str(X_test.shape)+ '\n'+
                   str(X_test.columns) + "\n" + 'Start ' + 
                   str(X_test.index[0]) + '\n' + 'End ' + str(X_test.index[-1])
                   + "\n\n y_test\n" + "=================================\n" +
                   'Number of (Records, Variables)' + str(y_test.shape)+ '\n'+
                   'Start ' + str(y_test.index[0]) + '\n' + 'End ' + 
                   str(y_test.index[-1]))
    
    X_train.to_csv(export_file_directory + station + '_X_train_original.csv',
                             header=True, index=True)
    y_train.to_csv(export_file_directory + station + '_y_train_original.csv',
                             header=True, index=True)
    X_test.to_csv(export_file_directory + station + '_X_test_original.csv',
                             header=True, index=True)
    y_test.to_csv(export_file_directory + station + '_y_test_original.csv',
                             header=True, index=True)
    print("Splitted Files created successfully for " + station)
    
    # Fitting and Transforming Feature Scaler
    feature_scaler = MinMaxScaler()
    feature_scaler_model = feature_scaler.fit(X_train)
    X_train_scaled = feature_scaler_model.transform(X_train)
    X_test_scaled = feature_scaler_model.transform(X_test)
    feature_scalar_params = feature_scaler_model.get_params()
    # Fitting and Transforming Target Scaler
    target_scaler = MinMaxScaler()
    y_train = (np.array(y_train)).reshape(-1, 1)
    y_test = (np.array(y_test)).reshape(-1, 1)
    target_scaler_model = target_scaler.fit(y_train)
    y_train_scaled = target_scaler_model.transform(y_train)
    y_test_scaled = target_scaler_model.transform(y_test)
    target_scalar_params = target_scaler_model.get_params()
    # Exporting Scaled Data
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, 
                                  index=X_train.index)
    X_train_scaled.to_csv(export_file_directory + station + '_X_train_scaled.csv',
                             header=True, index=True)
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=["rainfall_Tplus1",],
                                  index=X_train.index)
    y_train_scaled.to_csv(export_file_directory + station + '_y_train_scaled.csv',
                             header=True, index=True)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns,
                                 index=X_test.index)
    X_test_scaled.to_csv(export_file_directory + station + '_X_test_scaled.csv',
                             header=True, index=True)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=["rainfall_Tplus1",],
                                 index=X_test.index)
    y_test_scaled.to_csv(export_file_directory + station + '_y_test_scaled.csv',
                             header=True, index=True)
    
    print("Scaled Files created successfully for " + station)
    
    
    
    
    
        
    
    

