# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""

### Converting 2D Data Matrices of all variables to 1D Data Vectors 

### The current data format is as follows:
### Year Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
### 1950 2.1  3.2  4.3  1.2  5.9  7.1  2.3  9.8  1.2  2.1  2.2  1.1
### 1951 1.1  2.2  5.3  9.2  6.9  8.1  6.3  4.8  3.2  2.1  1.2  0.1
### ...

### The output data format will be as follows:
### Timestamp  TimeseriesData
### 1/1/1950      2.1
### 1/2/1950      3.2
### 1/3/1950      4.3
### 1/4/1950      1.2
### ...



import pandas as pd
import numpy as np
from my_custom_functions import getting_files_of_selected_extensions, write_description_vector_dataframe 
from datetime import datetime

# Enter the list of file extensions to be included
extension = [".xlsx"]
# Enter the path of the directory of input files
directory_path = "../Data/0. RawMatrixData/"
# Getting all files of selected extensions in a directory
selected_files_list = getting_files_of_selected_extensions(directory_path, 
                                                          extension)

# Enter the directory of output files
export_directory = "../Data/1. CleanTimeseriesData/"
program_name = "code01_timeseries_data_matrix_to_clean_timeseries_data_vector.py"

selected_month_names_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
                             'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
                             'Nov': 11, 'Dec': 12}
selected_col_names_list = selected_month_names_dict.keys()
custom_nans = [' ', '*', '**', '***', '****', '*****', '******', '*******']

readme_file_name = export_directory + '/' + 'readme.txt'

with open(readme_file_name, 'w') as readme_file:
    datetime_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    readme_file.write("Created with the program: " + program_name + "\n" + 
               "Date and Time: " + datetime_string + "\n" + 
               "Author: Sujan Chandra Mondol \n")
    for matrix_datafile in selected_files_list: 
        
        # Importing the CSV and Excel files differently
        if extension[0]=='.csv': 
            matrix_dataframe = pd.read_csv(directory_path + '/' + matrix_datafile,
                                   header=0, index_col=0)
        elif extension[0]=='.xlsx' or extension[0]=='.xls': 
            matrix_dataframe = pd.read_excel(directory_path + '/' + matrix_datafile,
                                   header=0, index_col=0)
        
        # Keeping only the the columns containing monthly data
        check_col_names = matrix_dataframe.columns
        if selected_col_names_list==[]:
            print(check_col_names)
            selected_col_names_list = input(
                "Enter the list of months: ").strip('''[]'"''').split("', '")
        matrix_dataframe = matrix_dataframe[selected_col_names_list]
        
        # Creating the vector dataframe containing the timeseries data
        vector_dataframe = pd.Series(dtype='float64')
        new_file_name = matrix_datafile.replace(extension[0], '.csv')
        for ind in matrix_dataframe.index: 
            status = 'NOT_last_year'
            if ind==matrix_dataframe.index[-1]:
                status = 'last_year'
            for col_name in matrix_dataframe.columns:
                if matrix_dataframe[col_name][ind] in custom_nans:
                    if status=='last_year':
                        break
                    datavalue = np.nan
                else:
                    datavalue = matrix_dataframe[col_name][ind]
                vector_dataframe[pd.Timestamp(year=ind, 
                                              month=selected_month_names_dict[col_name],
                                              day=1)
                                 ] = datavalue
                
        # Exporting the created vector dataframe into a CSV file
        vector_dataframe.to_csv(export_directory + '/' + new_file_name, header=False)
        print('The file [[', new_file_name, ']] has been created successfully...\n')
        
        # Writing Description in the ReadMe File
        heading = matrix_datafile
        write_description_vector_dataframe(vector_dataframe, heading, readme_file)
            
print("Program finished running. [OK]")
            
            

