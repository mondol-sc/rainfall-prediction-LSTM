# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def getting_files_of_selected_extensions(directory_path, extension_list):
    ''' Getting all files with the extensions selected in a list from a
    particular file directory '''

    import os

    file_or_directory_list = os.listdir(directory_path)
    print("List of all files and directories:\n", file_or_directory_list, "\n")

    selected_files_list = list() # creating an empty list

    # Getting the list of all files with listed extensions only
    for file_or_directory_name in file_or_directory_list:
        for extension in extension_list:
            if file_or_directory_name.endswith(extension):
                selected_files_list.append(file_or_directory_name)

    print("List of selected files:\n", selected_files_list, "\n")
    return selected_files_list

def write_description_vector_dataframe(vector_dataframe, heading, readme_file):
    number_of_nans = vector_dataframe.isna().sum().sum()
    number_of_records = vector_dataframe.shape[0]
    percentage_of_nans = number_of_nans*100 / number_of_records
    description = "First Month: " + str(vector_dataframe.index[0]) + '\n' + (
        "Last Month: ") + str(vector_dataframe.index[-1]) + '\n' + str(
            vector_dataframe.describe()) + "\nNumber of NaNs: "+ (
                      str(number_of_nans)) + '\n' + "Number of records: "+(
                      str(number_of_records))+'\n' +'Percentage of NaNs: '+(
                      str("%.2f" % percentage_of_nans)) + "%"
    readme_file.write("\n" + heading + "\n"
                      + "===================================================="
                      + "\n" + description + '\n\n')
def data_prep_to_feed_network(X_df, y_df, num_of_lookback_samples):
    X, y = np.array(X_df), np.array(y_df)
    X_new= list()
    y_new = list()
    for idx in range(num_of_lookback_samples, len(X)):
            X_new.append(X[(idx - num_of_lookback_samples):idx, :])
            y_new.append(y[idx,0])
    return np.array(X_new), np.array(y_new)


def actual_vs_predicted_timeseries_mini_plotter(actual, predicted, title,
                                                directory, prefix, figsize=(16,5)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(predicted.index, predicted, color = 'blue',
             label = 'Predicted Rainfall')
    plt.plot(predicted.index, actual, color = 'red', label = 'Actual Rainfall',
             linestyle="dashed")
    plt.title(title)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Rainfall (mm)', fontsize=12)
    plt.xlim(predicted.index.min(), predicted.index.max())
    ticks = list(predicted.index)
    plt.xticks(ticks, rotation=90)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    # Highlighting the time period from 15.12.2022 to 1.01.2023 and customizing the shaded area
    #plt.axvspan(datetime(2022, 12, 15), datetime(2023, 1, 1), facecolor='yellow', alpha=0.5, hatch='/', edgecolor='red', linewidth=5)
    plt.legend(loc='upper right', ncols=3, fontsize=12)
    plt.savefig(directory + prefix + 'TESTseries_lineplot.png')
    plt.show()

def actual_vs_predicted_timeseries_large_plotter(train, test, title, directory, prefix,
                                                 col_name="lstm_predicted", figsize=(16,5)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(train.index, train[col_name], color = 'blue',
             label = 'Predicted (Training)')
    plt.plot(test.index, test[col_name], color = 'magenta',
             label = 'Predicted (Testing)')
    ticks = list(np.hstack((train.index, test.index)))
    actual = np.hstack((train['actual'], test['actual']))
    plt.plot(ticks, actual, color = 'black', label = 'Actual',
             linestyle="dashed")
    plt.title(title)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Rainfall (mm)', fontsize=12)
    plt.xlim(ticks[0], ticks[-1])
    plt.xticks(ticks, rotation=90)
    plt.axvspan(test.index[0], test.index[-1], facecolor='lightgrey', alpha=0.1, hatch=None, edgecolor='black', linewidth=2)
    plt.tight_layout()
    plt.legend(loc='upper right', ncols=3, fontsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.savefig(directory + prefix + 'timeseries_lineplot_' + col_name + '.png' )
    plt.show()
