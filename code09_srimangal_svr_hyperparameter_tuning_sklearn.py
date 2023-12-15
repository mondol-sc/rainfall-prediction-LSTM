# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
import tensorflow as tf
import scipy.stats as stats
from my_custom_functions import data_prep_to_feed_network, actual_vs_predicted_timeseries_large_plotter, actual_vs_predicted_timeseries_mini_plotter


# Setting the seed value for ensuring the reproducibility of the experiment
seed = 43
tf.random.set_seed(seed)
np.random.seed(seed)

# Set the name of the station
station = 'srimangal'

# Importing Data from Files
input_file_directory = "../Data/5. SplitScaledData/"

X_train_df = pd.read_csv(input_file_directory + station + '_X_train_scaled.csv',
                          header=0, index_col=0)
y_train_df = pd.read_csv(input_file_directory + station + '_y_train_scaled.csv',
                          header=0, index_col=0)
X_test_df = pd.read_csv(input_file_directory + station + '_X_test_scaled.csv',
                          header=0, index_col=0)
y_test_df = pd.read_csv(input_file_directory + station + '_y_test_scaled.csv',
                          header=0, index_col=0)
y_train_original = pd.read_csv(input_file_directory + station + '_y_train_original.csv',
                          header=0, index_col=0) # for reverse scaling & final DF
y_test_original = pd.read_csv(input_file_directory + station + '_y_test_original.csv',
                          header=0, index_col=0) # for final dataframe



# Defining export directory
export_file_directory = "../Data/6. SrimangalStationResults/"
export_file_prefix = station + '_svr_'

# Preparing input data in required format
X_train = np.array(X_train_df)
X_test = np.array(X_test_df)
y_train = np.array(y_train_df).flatten()
y_test = np.array(y_test_df).flatten()


def hyperparameter_tuning(X_train, y_train, search_num, attempt_no):
    parameter_dists = {'kernel': ['poly', 'rbf', 'sigmoid'],
                       'C': stats.loguniform(1e-3, 1e3),
                       'gamma': stats.loguniform(1e-4,1)}

    svm_regressor = SVR()
    random_search = RandomizedSearchCV(estimator=svm_regressor,
                                       param_distributions=parameter_dists,
                                       scoring='neg_root_mean_squared_error',
                                       refit=True,
                                       n_iter=search_num,
                                       cv=4,
                                       verbose=4)
    random_search.fit(X_train, y_train)

    # Show best model description
    description = ("Best Regressor with Parameters:" + str(random_search.best_estimator_) + '\n\n' +
          "with score of," + str(random_search.best_score_) + '\n\n and with parameters: ' +
          str(random_search.best_params_))
    print(description)

    # Write the description for the best model
    best_model_description = open(export_file_directory + export_file_prefix +
                                  'best_model_description_' + str(attempt_no)+
                                  '.txt', 'w')
    best_model_description.write(description)
    best_model_description.close()

    random_search_results = pd.DataFrame(random_search.cv_results_)
    random_search_results = random_search_results.sort_values("rank_test_score")
    random_search_results.to_csv(export_file_directory + export_file_prefix +
                              'hyperparameter_tuning_' +
                              str(attempt_no) + '.csv')
    print("================================Tuning Completed================================")

    return random_search

tuning_results = hyperparameter_tuning(X_train, y_train, 3000, 1)
best_model_params = tuning_results.best_params_
tuned_model = SVR(kernel = best_model_params['kernel'],
                  gamma=best_model_params['gamma'],
                  C=best_model_params['C']
                  )

# Choose the  tuned model or a custom model as the selected model
selected_model = tuned_model.fit(X_train, y_train)  # custom_model or tuned_model

# Use the chosen model to predict based on both the test and training features
y_test_pred = selected_model.predict(X_test)
y_train_pred = selected_model.predict(X_train)

# Scaler Model Fitting
target_scaler = MinMaxScaler()
y_train_for_scaler = (np.array(y_train_original['rainfall_Tplus1'][:])).reshape(-1, 1)
target_scaler_model = target_scaler.fit(y_train_for_scaler)
# Reverse scaling of predicted data
y_train_PRED = target_scaler_model.inverse_transform(y_train_pred.reshape(1, -1)).flatten()
y_test_PRED = target_scaler_model.inverse_transform(y_test_pred.reshape(1, -1)).flatten()

# Elimination of negative values
for idx, element in enumerate(y_train_PRED):
    if element < 0:
        y_train_PRED[idx] = 0
for idx, element in enumerate(y_test_PRED):
    if element < 0:
        y_test_PRED[idx] = 0

# Create Final Dataframe with all actual and predicted target values for Training
target_train_dataframe = y_train_original.rename(columns={"rainfall_Tplus1": "actual"}, inplace=False)
target_train_dataframe['svr_predicted'] = [np.nan for i in range(target_train_dataframe.shape[0])]
target_train_dataframe.iloc[:, 1] = y_train_PRED.flatten()
target_train_dataframe.to_csv(export_file_directory + station +
                              '_svr_target_train_dataframe.csv', index=True,
                              header=True)


# Create Final Dataframe with all actual and predicted target values for Testing
target_test_dataframe = y_test_original.rename(columns={"rainfall_Tplus1": "actual"}, inplace=False)
target_test_dataframe['svr_predicted'] = [np.nan for i in range(target_test_dataframe.shape[0])]
target_test_dataframe.iloc[:, 1] = y_test_PRED.flatten()
target_test_dataframe.to_csv(export_file_directory + station +
                              '_svr_target_test_dataframe.csv', index=True,
                              header=True)
