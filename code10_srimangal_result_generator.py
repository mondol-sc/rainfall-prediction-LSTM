# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Declare Necessary Functions
def nash_sutcliffe_efficiency(y_actual, y_predicted):
    pred_error_variance = np.sum((y_actual - y_predicted) ** 2, axis=0)
    actual_variance = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return round(1 - (pred_error_variance/actual_variance), 5)

def coefficient_of_determination(y_actual, y_predicted):
    return round((pearsonr(y_actual, y_predicted)[0])**2, 5)

def root_mean_squared_error(y_actual, y_predicted):
    return round(mean_squared_error(y_actual, y_predicted, squared=False), 5)

def mae(y_actual, y_predicted):
    return round(mean_absolute_error(y_actual, y_predicted), 5)

def mse(y_actual, y_predicted):
    return round(mean_squared_error(y_actual, y_predicted), 5)



input_file_directory = "../Data/6. SrimangalStationResults/"
station = 'srimangal'
model_names = ['lstm', 'mlp', 'svr']

metrics_dataframe = pd.DataFrame(columns=model_names)

# Importing Data Files and Computing the values of Evaluation Metrics
for model in model_names:
    train_file_name = station + '_' + model + '_target_train_dataframe.csv'
    test_file_name = station + '_' + model + '_target_test_dataframe.csv'
    vars()['train_' + model] = pd.read_csv(input_file_directory + train_file_name,
                                          index_col=0,
                                          header=0, parse_dates=True)
    vars()['test_' + model] = pd.read_csv(input_file_directory + test_file_name,
                                          index_col=0,
                                          header=0, parse_dates=True)
    if model=='lstm':
        start=12
    else:
        start=0
    train_df = vars()['train_' + model]
    metrics_dataframe.loc['train_mae', model] = mae(train_df['actual'][start:], train_df.iloc[start:, 1])
    metrics_dataframe.loc['train_mse', model] = mse(train_df['actual'][start:], train_df.iloc[start:, 1])
    metrics_dataframe.loc['train_rmse', model] = root_mean_squared_error(train_df['actual'][start:], train_df.iloc[start:, 1])
    metrics_dataframe.loc['train_nse', model] = nash_sutcliffe_efficiency(train_df['actual'][start:], train_df.iloc[start:, 1])
    metrics_dataframe.loc['train_r2', model] = coefficient_of_determination(train_df['actual'][start:], train_df.iloc[start:, 1])

    test_df = vars()['test_' + model]
    metrics_dataframe.loc['test_mae', model] = mae(test_df['actual'][start:], test_df.iloc[start:, 1])
    metrics_dataframe.loc['test_mse', model] = mse(test_df['actual'][start:], test_df.iloc[start:, 1])
    metrics_dataframe.loc['test_rmse', model]= root_mean_squared_error(test_df['actual'][start:], test_df.iloc[start:, 1])
    metrics_dataframe.loc['test_nse', model] = nash_sutcliffe_efficiency(test_df['actual'][start:], test_df.iloc[start:, 1])
    metrics_dataframe.loc['test_r2', model]= coefficient_of_determination(test_df['actual'][start:], test_df.iloc[start:, 1])
metrics_dataframe.to_csv(input_file_directory + station + '_evaluation_metrics_table.csv', header=True, index_label='Model')


# Timeseries Plot: Actual and Predicted (LSTM, MLP, SVR)
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey = True, figsize=(12,6))
ticks = test_lstm.index
ax[0].plot(ticks, test_lstm['lstm_predicted'], color = 'black',
         label = 'LSTM Predicted', linestyle='dashed')
ax[0].plot(ticks, test_lstm['actual'], color = '#484848', label = 'Actual', 
         linestyle="solid")
ax[0].legend(loc='upper right', fontsize=12, ncol=2)
ax[1].plot(ticks, test_mlp['mlp_predicted'], color = 'black',
         label = 'MLP Predicted', linestyle='dashed')
ax[1].plot(ticks, test_mlp['actual'], color = '#484848', label = 'Actual',
         linestyle="solid")
ax[1].legend(loc='upper right', fontsize=12, ncol=2)
ax[1].set_ylabel(r'$\bf{Rainfall_{(t+1)}}$ (mm)', fontsize=12, loc='center', fontweight='bold')
ax[2].plot(ticks, test_svr['svr_predicted'], color = 'black',
         label = 'SVR Predicted', linestyle='dashed')
ax[2].plot(ticks, test_svr['actual'], color = '#484848', label = 'Actual',
         linestyle="solid")
ax[2].legend(loc='upper right', fontsize=12, ncol=2)
plt.xlabel('Time', fontsize=12, fontweight='bold')
plt.xlim(ticks[0], ticks[-1])
plt.xticks(ticks, rotation=0)
ax[2].xaxis.set_major_locator(mdates.YearLocator(1))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
fig.savefig(input_file_directory + station + '_test_timeseries_lineplots.png' )
plt.show()

# Actual vs Predcited Scatterplot for LSTM
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
actual = train_lstm['actual']
ax[0].scatter(actual[12:], train_lstm['lstm_predicted'][12:], color = '#303030', marker='.')
ax[0].plot(actual, actual, color = '#303030', label = 'Perfect Prediction', linestyle='dotted')
m,c = np.polyfit(actual[12:], train_lstm['lstm_predicted'][12:], 1)
ax[0].plot(actual, m*actual+c, color = 'black', label = 'Training Prediction', linestyle='solid')
ax[0].set_xlabel('Actual Rainfall (mm)', fontsize=12)
ax[0].set_ylabel('Predicted Rainfall (mm)', fontsize=12)
ax[0].legend(fontsize=11)
#----------------------------------------------------
actual = test_lstm['actual']
ax[1].scatter(actual[12:], test_lstm['lstm_predicted'][12:], color = '#303030', marker='.')
ax[1].plot(actual, actual, color = '#303030', label = 'Perfect Prediction', linestyle='dotted')
m,c = np.polyfit(actual[12:], test_lstm['lstm_predicted'][12:], 1)
ax[1].plot(actual, m*actual+c, color = 'black', label = 'Testing Prediction', linestyle='solid')
ax[1].set_xlabel('Actual Rainfall (mm)', fontsize=12)
ax[1].legend(fontsize=11)
fig.savefig(input_file_directory + station + '_LSTM_actual_vs_predicted_plot.png' )
plt.show()

# Actual vs Predcited Scatterplot for MLP
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
actual = train_mlp['actual']
ax[0].scatter(actual[12:], train_mlp['mlp_predicted'][12:], color = '#303030', marker='.')
ax[0].plot(actual, actual, color = '#303030', label = 'Perfect Prediction', linestyle='dotted')
m,c = np.polyfit(actual[12:], train_mlp['mlp_predicted'][12:], 1)
ax[0].plot(actual, m*actual+c, color = 'black', label = 'Training Prediction', linestyle='solid')
ax[0].set_xlabel('Actual Rainfall (mm)', fontsize=12)
ax[0].set_ylabel('Predicted Rainfall (mm)', fontsize=12)
ax[0].legend(fontsize=11)
#----------------------------------------------------
actual = test_mlp['actual']
ax[1].scatter(actual[12:], test_mlp['mlp_predicted'][12:], color = '#303030', marker='.')
ax[1].plot(actual, actual, color = '#303030', label = 'Perfect Prediction', linestyle='dotted')
m,c = np.polyfit(actual[12:], test_mlp['mlp_predicted'][12:], 1)
ax[1].plot(actual, m*actual+c, color = 'black', label = 'Testing Prediction', linestyle='solid')
ax[1].set_xlabel('Actual Rainfall (mm)', fontsize=12)
ax[1].legend(fontsize=11)
fig.savefig(input_file_directory + station + '_MLP_actual_vs_predicted_plot.png' )
plt.show()

# Actual vs Predcited Scatterplot for SVR
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
actual = train_svr['actual']
ax[0].scatter(actual[12:], train_svr['svr_predicted'][12:], color = '#303030', marker='.')
ax[0].plot(actual, actual, color = '#303030', label = 'Perfect Prediction', linestyle='dotted')
m,c = np.polyfit(actual[12:], train_svr['svr_predicted'][12:], 1)
ax[0].plot(actual, m*actual+c, color = 'black', label = 'Training Prediction', linestyle='solid')
ax[0].set_xlabel('Actual Rainfall (mm)', fontsize=12)
ax[0].set_ylabel('Predicted Rainfall (mm)', fontsize=12)
ax[0].legend(fontsize=11)
#----------------------------------------------------
actual = test_svr['actual']
ax[1].scatter(actual[12:], test_svr['svr_predicted'][12:], color = '#303030', marker='.')
ax[1].plot(actual, actual, color = '#303030', label = 'Perfect Prediction', linestyle='dotted')
m,c = np.polyfit(actual[12:], test_svr['svr_predicted'][12:], 1)
ax[1].plot(actual, m*actual+c, color = 'black', label = 'Testing Prediction', linestyle='solid')
ax[1].set_xlabel('Actual Rainfall (mm)', fontsize=12)
ax[1].legend(fontsize=11)
fig.savefig(input_file_directory + station + '_SVR_actual_vs_predicted_plot.png' )
plt.show()
