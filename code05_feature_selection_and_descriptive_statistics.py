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

input_file_directory = "../Data/3. ImputedData/"
export_file_directory = "../Data/4. FeatureSelectedDataWithDescriptions/"
program_name = 'code05_correlation_matrix_descriptive_statistics.py'

stations = ['sylhet', 'srimangal']

for station in stations:
    file_name = station + '_imputed_dataframe.csv' 
    tabular_dataframe = pd.read_csv(input_file_directory + file_name, 
                                    header=0, index_col=0) 
    tabular_dataframe.index = [pd.Timestamp(idx) for idx in tabular_dataframe.index]
    
    # Generate Pearson's Correlation Matrix
    corr_matrix = tabular_dataframe.corr(method="pearson")
    corr_matrix.to_csv(export_file_directory + station + '_corr_matrix.csv',
                             header=True, index=True)
    # Generate Correlation Heat Map Figure
    figure = sns.clustermap(corr_matrix, annot=True, linewidth=0.2, 
                            cmap=sns.diverging_palette(20, 220, n=200))
    
    plt.setp(figure.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(figure.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.show()
    # Export Correlation Heatmap Image
    figure.savefig(export_file_directory + station + "_correlation_heatmap.png") 
    
    
    readme_file_name = export_file_directory + station +  "_readme.txt"
    with open(readme_file_name, 'w') as readme_file:
        datetime_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        readme_file.write("Created with the program: " + program_name + "\n" + 
                   "Date and Time: " + datetime_string + "\n" + 
                   "Author: Sujan Chandra Mondol\n")
        removal_dict = dict()
        for variable in corr_matrix.columns:
            if np.abs(corr_matrix[variable]['rainfall_Tplus1']) < 0.1:
                removal_dict[variable] = corr_matrix[variable]['total_rainfall']
        removal_list = list(removal_dict.keys())
        readme_file.write("\n" + "Removed Feature Variables\n" 
                          + "====================================================\n"
                          + str(removal_dict) + '\n\n')
        print("The variables namely [[" + str(removal_list)+ "]] has been removed for station", station)
        custom_removal_list_str = input("Custom removed variables for " + station + ": ")
        custom_removal_list = custom_removal_list_str.split(',')
        readme_file.write("\n" + "Custom Removed Feature Variables\n" 
                          + "====================================================\n"
                          + str(custom_removal_list) + '\n\n')
        
    reduced_dataframe = tabular_dataframe.drop(removal_list, axis=1)
    reduced_dataframe = reduced_dataframe.drop(custom_removal_list, axis=1)
    # Exporting the Feature Selected Reduced Dataframe
    reduced_dataframe.to_csv(export_file_directory + station + '_reduced_dataframe.csv',
                             header=True, index=True)

    # Descriptive Statistics Table Generation
    descriptive_stats_table = (reduced_dataframe.describe()).transpose()
    # Exporting Descriptive Statistics Table 
    descriptive_stats_table.to_csv(export_file_directory + station + '_descriptive_stats.csv',
                             header=True, index=True)
    
    
    
    
    
