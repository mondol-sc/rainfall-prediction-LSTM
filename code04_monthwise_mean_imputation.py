# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_custom_functions import write_description_vector_dataframe
from datetime import datetime

input_file_directory = "../Data/2. ExtractedTabulatedData/"
export_file_directory = "../Data/3. ImputedData/"
program_name = 'code04_monthwise_mean_imputation.py'

stations = ['sylhet', 'srimangal']

for station in stations:
    file_name = station + '_extracted_tabulated_dataframe.csv' 
    tabular_dataframe = pd.read_csv(input_file_directory + file_name, 
                                    header=0, index_col=0) 
    tabular_dataframe.index = [pd.Timestamp(idx) for idx in tabular_dataframe.index]
    
    readme_file_name = export_file_directory + station +  "_readme.txt"
    with open(readme_file_name, 'w') as readme_file:
        datetime_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        readme_file.write("Created with the program: " + program_name + "\n" + 
                   "Date and Time: " + datetime_string + "\n" + 
                   "Author: Sujan Chandra Mondol\n")
        
        for variable in tabular_dataframe.columns:
            monthwise_means = {
            1: (tabular_dataframe[variable][tabular_dataframe.index.month == 1]).mean(),
            2: (tabular_dataframe[variable][tabular_dataframe.index.month == 2]).mean(),
            3: (tabular_dataframe[variable][tabular_dataframe.index.month == 3]).mean(),
            4: (tabular_dataframe[variable][tabular_dataframe.index.month == 4]).mean(),
            5: (tabular_dataframe[variable][tabular_dataframe.index.month == 5]).mean(),
            6: (tabular_dataframe[variable][tabular_dataframe.index.month == 6]).mean(),
            7: (tabular_dataframe[variable][tabular_dataframe.index.month == 7]).mean(),
            8: (tabular_dataframe[variable][tabular_dataframe.index.month == 8]).mean(),
            9: (tabular_dataframe[variable][tabular_dataframe.index.month == 9]).mean(),
            10: (tabular_dataframe[variable][tabular_dataframe.index.month == 10]).mean(),
            11: (tabular_dataframe[variable][tabular_dataframe.index.month == 11]).mean(),
            12: (tabular_dataframe[variable][tabular_dataframe.index.month == 12]).mean()}
            
            #Imputing with monthwise mean
            for idx in tabular_dataframe.index:
                if np.isnan(tabular_dataframe[variable][idx])==True:
                    month = (pd.Timestamp(idx)).month
                    tabular_dataframe[variable][idx] = monthwise_means[month]
            heading = station + "_" + variable
            write_description_vector_dataframe(tabular_dataframe[variable], heading, readme_file)
            print("Imputation completed successfully for " + heading + "...")
    
    # Populating Output Columns by Shifting
    tabular_dataframe['rainfall_Tplus1'] = tabular_dataframe['total_rainfall'].shift(-1)
    tabular_dataframe = tabular_dataframe[:][:-1]
    
    export_filename = station + "_imputed_dataframe.csv"
    tabular_dataframe.to_csv(export_file_directory + export_filename,
                             header=True, index=True)
    print("Imputed file [[" + export_filename + "]] created successfully///////")

