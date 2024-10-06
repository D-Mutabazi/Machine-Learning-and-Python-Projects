import torch
import talib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score


################################## Functions ####################################

# Data loading
def data_loader(filepath):
    data = pd.read_csv(filepath)
    data['Time'] = pd.to_datetime(data['Time'],format='%Y-%m-%d %H:%M:%S')
    data.set_index('Time', inplace=True)

    return data

# Feature Engineering 
def multivariateFeatureEngineering(data):
    
    #Trend following Indicators:

    #SMA - identofy long term trend
    data['50_sma'] = data['Close'].rolling(window=50).mean() 
    data['200_sma'] = data['Close'].rolling(window=200).mean() 

    #EMA - trend analysis: more weight applied to recent points
    data['50_ema'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['100_ema'] = data['Close'].ewm(span=100, adjust=False).mean()

    #MACD
    data['12_ema'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['26_ema'] = data['Close'].ewm(span=26, adjust=False).mean()

    data['MACD_line'] = data['12_ema']-data['26_ema'] # calculate the MACD line
    data['Signal_line'] = data['MACD_line'].ewm(span=9, adjust=False).mean() # 9-preiod ema signal calculated from the Macdline
    # data['MACD_histogram'] = data['MACD_line'] - data['Signal_line']

    #ADX
    # Calculate ADX using TA-Lib (14-period by default)
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

    #Momentum indicators:

    #RSI - 14-period
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    
    #Stochastic Oscillator
    data['stoch_k'], data['stoch_d'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)

    #Volatility indicators#:

    #ATR -Default period for ATR is 14
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    data = data.dropna() # drop rows that have NA

    #drop certain featires
    data = data.drop(columns=['12_ema', '26_ema'])

    return data


# Create Lag features and Multiple output response
def multivariateFeatureLagMultiStep(data, n_past, future_steps, target_column):
    features = []
    response = []

    max_future_step = max(future_steps)
    num_features = data.shape[1]
    group_feature_lags =  1 # change grouping of lagged features

    # Adjust the loop to prevent index out of bounds
    for i in range(n_past, len(data) - max_future_step + 1):

        if group_feature_lags==1:
                
            lagged_features = []

            for feature_idx in range(num_features):
                feature_lags = data.iloc[i - n_past:i, feature_idx].values 
                lagged_features.extend(feature_lags) 

        elif group_feature_lags==0:
            features.append(data.iloc[i - n_past:i, :].values)  # Take all columns as features


        if group_feature_lags==1:
            features.append(lagged_features)

        # Extract the target values at specified future steps using .iloc
        response.append([data.iloc[i + step - 1, target_column] for step in future_steps])

    # Convert lists to NumPy arrays after the loop
    features = np.array(features)  # Shape: (num_samples, n_past, num_features)
    response = np.array(response)  # Shape: (num_samples, len(future_steps))

    # Flatten the features to 2D array: (num_samples, n_past * num_features)
    features_flat = features.reshape(features.shape[0], -1)

    return features_flat, response


# Unique feature combinations
def featuresComblist(features):
    import itertools

    initial_feature = ['Close'] # Starting with the closing price

    # Get all combinations of the features list and add to the initial feature (Closing Price)
    feature_combinations = []
    for i in range(len(features) + 1):
        for combination in itertools.combinations(features, i):
            feature_combinations.append(list(combination)+ initial_feature )
    
    return feature_combinations


def initialize_csv(file_name):
    headers = ['lookback_window', 'feature_under_consideration', 'candidate_features', 'current_best_features', 
               'learning_rate', 'number_of_hidden_layers', 'number_of_hidden_neurons',
               'MSE_1_day', 'MAE_1_day', 'MAPE_1_day', 'MBE_1_day', 'RMSE_1_day', 'R2_1_day',
               'MSE_3_day', 'MAE_3_day', 'MAPE_3_day', 'MBE_3_day', 'RMSE_3_day', 'R2_3_day',
               'MSE_5_day', 'MAE_5_day', 'MAPE_5_day', 'MBE_5_day', 'RMSE_5_day', 'R2_5_day']
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_name, index=False)
# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    # Ensure y_true and y_pred are PyTorch tensors for MSE calculation
    mse = F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true)).item()
    
    # Convert to NumPy arrays for the remaining metrics
    y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE
    mbe = np.mean(y_true - y_pred)  # Mean Bias Error (MBE)
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(y_true, y_pred)  # R²
    return mse, mae, mape, mbe, rmse, r2

# Function to append results to CSV (includes candidate_features and current_best_features)
def append_to_csv(file_name, hyperparams, metrics, candidate_features, current_best_features, feature_under_consideration):
    # Convert lists of features to strings to store them in the CSV
    # candidate_features_str = ', '.join(candidate_features)
    # current_best_features_str = ', '.join(current_best_features)
    
    # Append the feature under consideration, candidate, and best features to hyperparameters list
    row = [hyperparams[0], feature_under_consideration, candidate_features, current_best_features] + hyperparams[1:] + metrics
    
    df = pd.DataFrame([row])
    df.to_csv(file_name, mode='a', header=False, index=False)

########################################### Model training ##########################################################


class multipleOutputForexDataset:
    def __init__(self, data, n_past, futureSteps, target_col):
        self.features = []
        self.labels = []

        self.data = data

        #creat design matix and response
        self.features, self.labels = multivariateFeatureLagMultiStep(self.data, n_past, futureSteps, target_col)

    def __len__(self):
        return len(self.features)
    

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)  # Change to float32
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)  # Ensure both are float32
        return features, labels


######################################################## Data Loading and default values ################################
file_path='../Data/EURUSD_D1.csv'
FXdata = data_loader(file_path)

future_steps = [1 , 3, 5]
target_col =  -1


########################################### Neural Network model and Architecture development ##############################


# Define the hyperparameter grid
OG_features = ['Open', 'High', 'Low', 'Volume', '50_sma', '200_sma', '50_ema',
       '100_ema', 'MACD_line', 'Signal_line', 'ADX', 'RSI', 'stoch_k',
       'stoch_d', 'ATR']
output_size = 3  # Output layer size (closing price t =  [1 , 3, 5])

lookback_window_grid = [1, 3, 5, 7, 10]  
hidden_neurons_grid = [8 ,10, 16, 32, 64, 128]  # Number of neurons per hidden layer
hidden_layers_grid = [1, 2, 3, 4]  # Number of hidden layers
learning_rate_grid = [0.001]  # Learning rates


#Generate initial features
multiVarData = multivariateFeatureEngineering(FXdata)
col = [col for col in multiVarData.columns if col!='Close'] + ['Close'] #Arrange the columns
multiVarData=   multiVarData[col]
               
# Forward selection features
best_features = ['Close']
best_score = float('inf')  # Initialize best score

import time

# Record the start time
# start_time = time.time()

# Initialize CSV file
file_name = 'NN_hyperparameter_search_results.csv'
initialize_csv(file_name)
# feature_names_used = list(features)  # Store this for later use


for n_past in lookback_window_grid:

    for hidden_size in hidden_neurons_grid:

        for num_layers in hidden_layers_grid:

            for learning_rate in learning_rate_grid:

            
                ############################################## Data Loading for Closing Feature #################################
                data_subset = (multiVarData['Close']).to_frame() # Select initial start feature>> convert 1d series tro 2d

                # Prepare the dataset for the current lookback window
                transformed_data = multipleOutputForexDataset(data_subset, n_past, future_steps, target_col) # returns lagged features

                # Train-test split
                train_dataset, test_dataset = train_test_split(transformed_data, test_size=0.2, random_state=42, shuffle=False)

                # Load train and test data
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


                ######################################## Define Initial Model Architecture ###########################################


                # Initialize weights and biases for the layers
                W_hidden = []
                b_hidden = []
                input_size = data_subset.shape[1]  # number of original features


                # First layer initialization
                W_hidden.append(torch.randn(input_size * n_past, hidden_size, dtype=torch.float32, requires_grad=True))
                b_hidden.append(torch.randn(hidden_size, dtype=torch.float32, requires_grad=True))

                # Initialize additional hidden layers if required
                for i in range(1, num_layers):
                    W_hidden.append(torch.randn(hidden_size, hidden_size, dtype=torch.float32, requires_grad=True))
                    b_hidden.append(torch.randn(hidden_size, dtype=torch.float32, requires_grad=True))

                # Output layer initialization
                W_output = torch.randn(hidden_size, output_size, dtype=torch.float32, requires_grad=True)
                b_output = torch.randn(output_size, dtype=torch.float32, requires_grad=True)

                # Define forward propagation
                def forward(features):
                    # Input to first hidden layer
                    hidden_activation = F.relu(features @ W_hidden[0] + b_hidden[0])

                    # Forward through remaining hidden layers
                    for i in range(1, num_layers):
                        hidden_activation = F.relu(hidden_activation @ W_hidden[i] + b_hidden[i])

                    # Output layer
                    output = hidden_activation @ W_output + b_output

                    # Check for NaN values and replace them with 0
                    output = torch.nan_to_num(output, nan=0.0)

                    # Limit the output values between -50 and 50
                    output = torch.clamp(output, min=-50.0, max=50.0)

                    return output

                ########################################## Training the model##################################################

                num_epochs = 100

                for epoch in range(num_epochs):
                    for batch_idx, (features, labels) in enumerate(train_loader):
                        # Zero gradients
                        for i in range(num_layers):
                            W_hidden[i].grad = None
                            b_hidden[i].grad = None
                        W_output.grad = None
                        b_output.grad = None

                        # Forward pass
                        predicted_close = forward(features)


                        # Compute loss
                        loss = F.mse_loss(predicted_close, labels)

                        # Backward pass
                        loss.backward()

                        # Gradient descent
                        with torch.no_grad():
                            for i in range(num_layers):
                                W_hidden[i] -= learning_rate * W_hidden[i].grad
                                b_hidden[i] -= learning_rate * b_hidden[i].grad
                            W_output -= learning_rate * W_output.grad
                            b_output -= learning_rate * b_output.grad


                # Evaluate on the test data and compute metrics for each forecast horizon
                mse_1, mae_1, mape_1, mbe_1, rmse_1, r2_1 = 0, 0, 0, 0, 0, 0
                mse_3, mae_3, mape_3, mbe_3, rmse_3, r2_3 = 0, 0, 0, 0, 0, 0
                mse_5, mae_5, mape_5, mbe_5, rmse_5, r2_5 = 0, 0, 0, 0, 0, 0

                with torch.no_grad():
                    for features, labels in test_loader:
                        predicted_close = forward(features)

                        # Compute metrics for each forecast horizon (1-day, 3-day, 5-day)
                        for day_idx, day_name in zip(range(3), ['1_day', '3_day', '5_day']):
                            y_true = labels[:, day_idx].numpy()
                            y_pred = predicted_close[:, day_idx].numpy()

                            mse, mae, mape, mbe, rmse, r2 = calculate_metrics(y_true, y_pred)

                            # Assign to the correct metric based on the day
                            if day_name == '1_day':
                                mse_1, mae_1, mape_1, mbe_1, rmse_1, r2_1 = mse, mae, mape, mbe, rmse, r2
                            elif day_name == '3_day':
                                mse_3, mae_3, mape_3, mbe_3, rmse_3, r2_3 = mse, mae, mape, mbe, rmse, r2
                            elif day_name == '5_day':
                                mse_5, mae_5, mape_5, mbe_5, rmse_5, r2_5 = mse, mae, mape, mbe, rmse, r2

                
                average_mae = (mae_1 + mae_3 + mae_5) / 3

                best_score  = average_mae

                current_best_features = best_features.copy()  # Start with 'Close'

                feature_under_consideration = 'Close'  # before forward selection only close used

                candidate_features = 'Close'

                # Prepare hyperparameters and metrics
                hyperparams = [n_past, learning_rate, num_layers, hidden_size]
                metrics = [
                    mse_1, mae_1, mape_1, mbe_1, rmse_1, r2_1,
                    mse_3, mae_3, mape_3, mbe_3, rmse_3, r2_3,
                    mse_5, mae_5, mape_5, mbe_5, rmse_5, r2_5
                ]

                # Append the current results to the CSV, including the feature under consideration
                append_to_csv(file_name, hyperparams, metrics, candidate_features, current_best_features, feature_under_consideration)


                ###############################################################################################
                #                                          Peform Forward Selection                           #
                ############################################################################################### 
            
            
                for feature in OG_features:
                    # Add the new feature to the current best features and selectipn
                    candidate_features = current_best_features + [feature]

                    # Put close at the end of the list
                    if 'Close' in candidate_features:
                        candidate_features.remove('Close')

                    candidate_features.append('Close')

                        # Put close at the end of the list
                    if 'Close' in current_best_features:
                        current_best_features.remove('Close')

                    current_best_features.append('Close')

                    data_subset = multiVarData[candidate_features] # select subset of data

                    ######################################## Define Model Architecture based on tranformed Features ###########################################
                
                   
                    # Initialize weights and biases for the layers
                    W_hidden = []
                    b_hidden = []
                    input_size = data_subset.shape[1]  # number of original features

                    # First layer initialization
                    W_hidden.append(torch.randn(input_size * n_past, hidden_size, dtype=torch.float32, requires_grad=True))
                    b_hidden.append(torch.randn(hidden_size, dtype=torch.float32, requires_grad=True))

                    # Initialize additional hidden layers if required
                    for i in range(1, num_layers):
                        W_hidden.append(torch.randn(hidden_size, hidden_size, dtype=torch.float32, requires_grad=True))
                        b_hidden.append(torch.randn(hidden_size, dtype=torch.float32, requires_grad=True))

                    # Output layer initialization
                    W_output = torch.randn(hidden_size, output_size, dtype=torch.float32, requires_grad=True)
                    b_output = torch.randn(output_size, dtype=torch.float32, requires_grad=True)

                    # Define forward propagation
                    def forward(features):
                        # Input to first hidden layer
                        hidden_activation = F.relu(features @ W_hidden[0] + b_hidden[0])

                        # Forward through remaining hidden layers
                        for i in range(1, num_layers):
                            hidden_activation = F.relu(hidden_activation @ W_hidden[i] + b_hidden[i])

                        # Output layer
                        output = hidden_activation @ W_output + b_output

                        # Check for NaN values and replace them with 0
                        output = torch.nan_to_num(output, nan=0.0)

                        # Limit the output values between -50 and 50
                        output = torch.clamp(output, min=-50.0, max=50.0)

                        return output

                    ############################################## Data Loading #################################################

                    # Prepare the dataset for the current lookback window
                    transformed_data = multipleOutputForexDataset(data_subset, n_past, future_steps, target_col)

                    # Train-test split
                    train_dataset, test_dataset = train_test_split(transformed_data, test_size=0.2, random_state=42, shuffle=False)

                    # Load train and test data
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
                    ########################################## Training the model##################################################
                    num_epochs = 100

                    for epoch in range(num_epochs):
                        for batch_idx, (features, labels) in enumerate(train_loader):
                            # Zero gradients
                            for i in range(num_layers):
                                W_hidden[i].grad = None
                                b_hidden[i].grad = None
                            W_output.grad = None
                            b_output.grad = None

                            # Forward pass
                            predicted_close = forward(features)


                            # Compute loss
                            loss = F.mse_loss(predicted_close, labels)

                            # Backward pass
                            loss.backward()

                            # Gradient descent
                            with torch.no_grad():
                                for i in range(num_layers):
                                    W_hidden[i] -= learning_rate * W_hidden[i].grad
                                    b_hidden[i] -= learning_rate * b_hidden[i].grad
                                W_output -= learning_rate * W_output.grad
                                b_output -= learning_rate * b_output.grad

                
                    ########################################### Model Evaluation and Result recording #######################################################
                                            
                    # Evaluate on the test data and compute metrics for each forecast horizon
                    mse_1, mae_1, mape_1, mbe_1, rmse_1, r2_1 = 0, 0, 0, 0, 0, 0
                    mse_3, mae_3, mape_3, mbe_3, rmse_3, r2_3 = 0, 0, 0, 0, 0, 0
                    mse_5, mae_5, mape_5, mbe_5, rmse_5, r2_5 = 0, 0, 0, 0, 0, 0

                    with torch.no_grad():
                        for features, labels in test_loader:
                            predicted_close = forward(features)

                            # Compute metrics for each forecast horizon (1-day, 3-day, 5-day)
                            for day_idx, day_name in zip(range(3), ['1_day', '3_day', '5_day']):
                                y_true = labels[:, day_idx].numpy()
                                y_pred = predicted_close[:, day_idx].numpy()

                                mse, mae, mape, mbe, rmse, r2 = calculate_metrics(y_true, y_pred)

                                # Assign to the correct metric based on the day
                                if day_name == '1_day':
                                    mse_1, mae_1, mape_1, mbe_1, rmse_1, r2_1 = mse, mae, mape, mbe, rmse, r2
                                elif day_name == '3_day':
                                    mse_3, mae_3, mape_3, mbe_3, rmse_3, r2_3 = mse, mae, mape, mbe, rmse, r2
                                elif day_name == '5_day':
                                    mse_5, mae_5, mape_5, mbe_5, rmse_5, r2_5 = mse, mae, mape, mbe, rmse, r2


                    average_mae = (mae_1 + mae_3 + mae_5) / 3

                    mae_score = average_mae

                    # If the new feature improves performance, keep it
                    if  mae_score < best_score:
                        best_score = mae_score
                        current_best_features.append(feature)

                    # Prepare hyperparameters and metrics
                    hyperparams = [n_past, learning_rate, num_layers, hidden_size]
                    metrics = [
                        mse_1, mae_1, mape_1, mbe_1, rmse_1, r2_1,
                        mse_3, mae_3, mape_3, mbe_3, rmse_3, r2_3,
                        mse_5, mae_5, mape_5, mbe_5, rmse_5, r2_5
                    ]

                    # Append the current results to the CSV, including the feature under consideration
                    append_to_csv(file_name, hyperparams, metrics, candidate_features, current_best_features, feature)

                    # # Record the end time
                    # end_time = time.time()

                    # # Calculate the elapsed time
                    # elapsed_time = end_time - start_time

                    # print(f"Execution time for this set of parameters: {elapsed_time:.2f} seconds")


print("Grid search completed and results saved to CSV!")








