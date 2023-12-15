# -*- coding: utf-8 -*-
"""
@author: Sujan Chandra Mondol
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from my_custom_functions import data_prep_to_feed_network, actual_vs_predicted_timeseries_large_plotter, actual_vs_predicted_timeseries_mini_plotter

# Setting the seed value for ensuring the reproducibility of the experiment
seed = 43
tf.random.set_seed(seed)
np.random.seed(seed)

# Set the name of the station
station = 'sylhet'

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
export_file_directory = "../Data/7. SylhetStationResults/"
export_file_prefix = station + '_lstm_'

# Preparing Data as the model input
num_of_lookback_samples = 12
X_train, y_train =  data_prep_to_feed_network(X_train_df, y_train_df, num_of_lookback_samples)
X_test, y_test =  data_prep_to_feed_network(X_test_df, y_test_df, num_of_lookback_samples)
print(X_train.shape)

# Defining the model structure
def prediction_model_function(unit_num, learning_rate):
    opt = Adam(learning_rate = learning_rate)
    # Instatntiate model
    pred_model = Sequential()
    # Adding necessary layers
    pred_model.add(LSTM(unit_num,
                        input_shape = (num_of_lookback_samples,X_train.shape[2]),
                        return_sequences=True,
                        activation='tanh'))
    pred_model.add(Dropout(0.50))
    pred_model.add(BatchNormalization())
    pred_model.add(LSTM(units=unit_num,
                        return_sequences=False,
                        activation='tanh'))
    pred_model.add(Dropout(0.20))
    pred_model.add(BatchNormalization())
    pred_model.add(Dense(1))
    # Compile the model
    pred_model.compile(loss = 'mean_squared_error',optimizer = opt)
    # Return model setup
    return pred_model

# Setting up the hyperparameter tuning system
def hyperparameter_tuning(X_train, y_train, number_of_searches, attempt_no):
    # Calling the model function with KerasRegressor
    pred_model = KerasRegressor(build_fn=prediction_model_function,verbose=2,
                                validation_split=0.25)
    # Listing all different hyperparameters
    hyperparameters = {'unit_num': [4,8,12,16,32,64,128],
                       'batch_size' : [4,8,16,32,64,128,256],
                       'learning_rate':stats.loguniform(1e-4, 1e-1)}
    # Defining the grid search operation
    random_search  = RandomizedSearchCV(estimator = pred_model,
                                param_distributions = hyperparameters,
                                n_iter = number_of_searches,
                                cv = 4,
                                verbose=4)
    # Setting the patience value for early stopping criteria
    early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min',
                                           verbose=1, patience=50)

    # Run the hyperparameter tuning operation
    random_search_run = random_search.fit(X_train,y_train,
                                      callbacks=[early_stopping_monitor],
                                      epochs=2000)

    # Show tuning completetion message
    print("========================Tuning Completed========================")

    # Loss_vs_ep

    # Show the best model parameters
    print('\n\n Best Model Hyperparameters: ', random_search_run.best_params_)

    # Capturing and exporting all grid search models with results
    random_search_results = pd.DataFrame(random_search_run.cv_results_)
    random_search_results = random_search_results.sort_values("rank_test_score")
    random_search_results.to_csv(export_file_directory + export_file_prefix +
                              'hyperparameter_tuning_' +
                              str(attempt_no)+ '.csv')

    # Choose the best model as the selected model
    best_model = random_search_run.best_estimator_.model

    # Write the description for the best model
    best_model_description = open(export_file_directory + export_file_prefix +
                                  'best_model_description_' + str(attempt_no)+
                                  '.txt', 'w')
    description = str(random_search_run.best_params_) + "\n\n" + str(best_model.summary())
    best_model_description.write(description)
    best_model_description.close()

    # Visualize and export the model structure
    plot_model(best_model, to_file=export_file_directory + export_file_prefix +
               'best_model_plot_' + str(attempt_no) + '.png',
               show_shapes=True,)

    # Return the best model and corresponding history
    return best_model, random_search_run

# Get the model resulting from hyperparameter tuning
(tuned_model, tuning_history) = hyperparameter_tuning(X_train, y_train, 300, 1)

# Option for going with a custom model
'''
# For custom model
custom_model = prediction_model_function(32, 0.01)
early_stopping_monitor = EarlyStopping(patience=50)
model_history = custom_model.fit(X_train,y_train, epochs=2000, batch_size=256,
                                 callbacks=[early_stopping_monitor],
                                validation_split=0.5)
'''
# Choose the  tuned model or a custom model as the selected model
selected_model = tuned_model  # custom_model or tuned_model

# Use the chosen model to predict based on both the test and training features
y_test_pred = selected_model.predict(X_test)
y_train_pred = selected_model.predict(X_train)

# Scaler Model Fitting
target_scaler = MinMaxScaler()
y_train_for_scaler = (np.array(y_train_original['rainfall_Tplus1'][:])).reshape(-1, 1)
target_scaler_model = target_scaler.fit(y_train_for_scaler)
# Reverse scaling of predicted data
y_train_PRED = target_scaler_model.inverse_transform(y_train_pred)
y_test_PRED = target_scaler_model.inverse_transform(y_test_pred)

# Elimination of negative values
for idx, element in enumerate(y_train_PRED):
    if element < 0:
        y_train_PRED[idx] = 0
for idx, element in enumerate(y_test_PRED):
    if element < 0:
        y_test_PRED[idx] = 0

# Create Final Dataframe with all actual and predicted target values for Training
target_train_dataframe = y_train_original.rename(columns={"rainfall_Tplus1": "actual"}, inplace=False)
target_train_dataframe['lstm_predicted'] = [np.nan for i in range(target_train_dataframe.shape[0])]
target_train_dataframe.iloc[num_of_lookback_samples:, 1] = y_train_PRED.flatten()
target_train_dataframe.to_csv(export_file_directory + export_file_prefix +
                              'target_train_dataframe.csv', index=True,
                              header=True)


# Create Final Dataframe with all actual and predicted target values for Testing
target_test_dataframe = y_test_original.rename(columns={"rainfall_Tplus1": "actual"}, inplace=False)
target_test_dataframe['lstm_predicted'] = [np.nan for i in range(target_test_dataframe.shape[0])]
target_test_dataframe.iloc[num_of_lookback_samples:, 1] = y_test_PRED.flatten()
target_test_dataframe.to_csv(export_file_directory + export_file_prefix +
                              'target_test_dataframe.csv', index=True,
                              header=True)
