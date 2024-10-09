import torch
import talib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import time

################################## Ensure reproducility #############################

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# Using CUDA (GPU) operations and want to ensure full determinism:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # remove ? not using CNN??

################################ Measure Program execution time ##################
start_time  = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
def multivariateFeatureLagMultiStep(data, n_past, future_steps, target_column, scaler=None, fit_scaler=True):
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

    
    # Standardize the Features
    if fit_scaler:  # Fit the scaler only on the training+validation(because kept together) data
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:  # Use the pre-fitted scaler for validation/test data
        features = scaler.transform(features)


    # Flatten the features to 2D array: (num_samples, n_past * num_features)
    features_flat = features.reshape(features.shape[0], -1)

    return features_flat, response, scaler


def initialize_csv(file_name):
    headers = ['lookback_window', 'feature_under_consideration', 'candidate_features', 'current_best_features', 
               'learning_rate', 'number_of_hidden_layers', 'number_of_hidden_neurons','best_model_sw', 'best_model_hn','best_model_hl', 'best_model_mse','best_model_features',
               'MSE_1_day', 'MAE_1_day', 'MAPE_1_day', 'MBE_1_day', 'RMSE_1_day',
               'MSE_3_day', 'MAE_3_day', 'MAPE_3_day', 'MBE_3_day', 'RMSE_3_day', 
               'MSE_5_day', 'MAE_5_day', 'MAPE_5_day', 'MBE_5_day', 'RMSE_5_day']
    df = pd.DataFrame(columns=headers)
    df.to_csv(file_name, index=False)



# # Function to calculate performance metrics
# def calculate_metrics(y_true, y_pred):
#     # Ensure y_true and y_pred are PyTorch tensors for MSE calculation
#     mse = F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true)).item()
    
#     # Convert to NumPy arrays for the remaining metrics
#     y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
#     y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
#     mae = mean_absolute_error(y_true, y_pred)
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE
#     mbe = np.mean(y_true - y_pred)  # Mean Bias Error (MBE)
#     rmse = np.sqrt(mse)  # RMSE
#     r2 = r2_score(y_true, y_pred)  # R²
#     return mse, mae, mape, mbe, rmse, r2
    
def calculate_metrics(y_true, y_pred):
   
    mse = F.mse_loss(y_pred, y_true).item()

    # Use torch operations to compute metrics (keeping it on GPU)
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100  # MAPE
    mbe = torch.mean(y_true - y_pred).item()  # Mean Bias Error (MBE)
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()  # RMSE

    return mse, mae, mape, mbe, rmse


# Function to append results to CSV (includes candidate_features and current_best_features)
def append_to_csv(file_name, hyperparams, metrics, candidate_features, current_best_features, feature_under_consideration, best_model_info):
    # Convert lists of features to strings to store them in the CSV
    candidate_features_str = ', '.join(candidate_features)
    current_best_features_str = ', '.join(current_best_features)
    
    # Append the feature under consideration, candidate, and best features to hyperparameters list
    row = [hyperparams[0], feature_under_consideration, candidate_features_str, current_best_features_str] + hyperparams[1:] + best_model_info+ metrics
    
    df = pd.DataFrame([row])
    df.to_csv(file_name, mode='a', header=False, index=False)

# Function for data loading and splitting
def load_dataset(data, n_past, future_steps, target_col, batch_size=32, test_size=0.11103416435):

    design_matrix, labels, scaler  = multivariateFeatureLagMultiStep(data, n_past, future_steps,target_col )
    transformed_data = multipleOutputForexDataset(design_matrix, labels)     # Prepare dataset (training and validation) with lagging features that have been standardized

    # training - validation split
    train_dataset, validation_set = train_test_split(transformed_data, test_size=0.11103416435,random_state=42, shuffle=False )
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    
    return train_loader, validation_loader, scaler

# Function to initialize the model
import torch
import torch.nn as nn

# Function to initialize the model using He initialization
def initialize_model(input_size, hidden_size, num_layers, output_size, device):
    W_hidden = []
    b_hidden = []
    
    # Initialize hidden layers with He initialization (normal distribution)
    W_hidden.append(torch.nn.Parameter(torch.empty(input_size, hidden_size).to(device)))
    nn.init.kaiming_normal_(W_hidden[0], nonlinearity='relu')  # He initialization for ReLU
    
    b_hidden.append(torch.nn.Parameter(torch.zeros(hidden_size).to(device)))  # Bias initialized to zero
    
    for i in range(1, num_layers):
        W_hidden.append(torch.nn.Parameter(torch.empty(hidden_size, hidden_size).to(device)))
        nn.init.kaiming_normal_(W_hidden[i], nonlinearity='relu')  # He initialization for ReLU
        
        b_hidden.append(torch.nn.Parameter(torch.zeros(hidden_size).to(device)))  # Bias initialized to zero
    
    # Initialize output layer with He initialization
    W_output = torch.nn.Parameter(torch.empty(hidden_size, output_size).to(device))
    nn.init.kaiming_normal_(W_output, nonlinearity='relu')  # He initialization for output layer
    
    b_output = torch.nn.Parameter(torch.zeros(output_size).to(device))  # Bias initialized to zero
    
    # Return the model parameters
    return W_hidden, b_hidden, W_output, b_output

# Forward pass function
def forward_pass(features, W_hidden, b_hidden, W_output, b_output, num_layers):
    hidden_activation = F.relu(features @ W_hidden[0] + b_hidden[0])
    
    for i in range(1, num_layers):
        hidden_activation = F.relu(hidden_activation @ W_hidden[i] + b_hidden[i])
    
    output = hidden_activation @ W_output + b_output
    return torch.clamp(torch.nan_to_num(output, nan=0.0), min=-50.0, max=50.0)

# Training function with early stopping
def train_model(train_loader, val_loader, W_hidden, b_hidden, W_output, b_output, optimizer, device, num_epochs=2000, patience=20, min_delta=0.001):
    best_loss = float('inf')  # Initialize the best loss as infinity
    patience_counter = 0  # Counter to track how long we've gone without improvement

    for epoch in range(num_epochs):
        # Training loop
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_close = forward_pass(features, W_hidden, b_hidden, W_output, b_output, len(W_hidden))

            # Compute loss
            loss = F.mse_loss(predicted_close, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # After each epoch, evaluate on the validation set
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                predicted_close = forward_pass(features, W_hidden, b_hidden, W_output, b_output, len(W_hidden))
                val_loss += F.mse_loss(predicted_close, labels).item()

        val_loss /= len(val_loader)  # Average validation loss

        # Early stopping with min_delta
        if (best_loss - val_loss) > min_delta:  # Only count as an improvement if the difference is larger than min_delta
            best_loss = val_loss  # Update best loss
            patience_counter = 0  # Reset patience counter

            #store the weight of the improning model
        else:
            patience_counter += 1  # Increment patience counter if no significant improvement

        # # If patience counter exceeds the patience limit, stop training
        if patience_counter >= patience:
            # print(f"Early stopping at epoch {epoch+1}, validation loss did not improve for {patience} consecutive epochs.")
            break

        # # Optionally print loss for each epoch
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss:.6f}")

# Evaluation function - validation set
def evaluate_model(val_loader, W_hidden, b_hidden, W_output, b_output, num_layers, device):
    mse_1, mae_1, mape_1, mbe_1, rmse_1  = 0, 0, 0, 0, 0
    mse_3, mae_3, mape_3, mbe_3, rmse_3  = 0, 0, 0, 0, 0
    mse_5, mae_5, mape_5, mbe_5, rmse_5 = 0, 0, 0, 0, 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            
            predicted_close = forward_pass(features, W_hidden, b_hidden, W_output, b_output, num_layers)
            
            for day_idx, day_name in zip(range(3), ['1_day', '3_day', '5_day']):
                y_true = labels[:, day_idx]
                y_pred = predicted_close[:, day_idx]
                
                mse, mae, mape, mbe, rmse = calculate_metrics(y_true, y_pred)
                
                if day_name == '1_day':
                    mse_1, mae_1, mape_1, mbe_1, rmse_1 = mse, mae, mape, mbe, rmse
                elif day_name == '3_day':
                    mse_3, mae_3, mape_3, mbe_3, rmse_3 = mse, mae, mape, mbe, rmse
                elif day_name == '5_day':
                    mse_5, mae_5, mape_5, mbe_5, rmse_5 = mse, mae, mape, mbe, rmse
            
    # Evaluate average model performance

    return mse_1, mae_1, mape_1, mbe_1, rmse_1, mse_3, mae_3, mape_3, mbe_3, rmse_3, mse_5, mae_5, mape_5, mbe_5, rmse_5

# Evaluation function - test set
def test_set_evaluate_model(test_loader, num_layers, device, model_path='best_model_and_scaler.pth'):
    # Load the best model's weights and scaler used on best set from the saved file
    saved_data = torch.load(model_path)
    saved_model_weights = saved_data['model_weights']
    scaler = saved_data['scaler']

    # Unpack the saved model weights
    W_hidden = saved_model_weights['W_hidden']
    b_hidden = saved_model_weights['b_hidden']
    W_output = saved_model_weights['W_output']
    b_output = saved_model_weights['b_output']

    mse_1, mae_1, mape_1, mbe_1, rmse_1 = 0, 0, 0, 0, 0
    mse_3, mae_3, mape_3, mbe_3, rmse_3 = 0, 0, 0, 0, 0
    mse_5, mae_5, mape_5, mbe_5, rmse_5 = 0, 0, 0, 0, 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            predicted_close = forward_pass(features, W_hidden, b_hidden, W_output, b_output, num_layers)
            
            for day_idx, day_name in zip(range(3), ['1_day', '3_day', '5_day']):
                y_true = labels[:, day_idx]
                y_pred = predicted_close[:, day_idx]
                
                mse, mae, mape, mbe, rmse = calculate_metrics(y_true, y_pred)
                
                if day_name == '1_day':
                    mse_1, mae_1, mape_1, mbe_1, rmse_1 = mse, mae, mape, mbe, rmse
                elif day_name == '3_day':
                    mse_3, mae_3, mape_3, mbe_3, rmse_3 = mse, mae, mape, mbe, rmse
                elif day_name == '5_day':
                    mse_5, mae_5, mape_5, mbe_5, rmse_5 = mse, mae, mape, mbe, rmse

    return mse_1, mae_1, mape_1, mbe_1, rmse_1, mse_3, mae_3, mape_3, mbe_3, rmse_3, mse_5, mae_5, mape_5, mbe_5, rmse_5



# Main function to run the training and evaluation
def run_training(data, learning_rate, n_past, future_steps, target_col, hidden_size, num_layers, output_size,current_best_score,  device):

    best_model_weights = {} # Dictionary to store the best models weigts
    best_scaler =  None

    train_loader,validation_loader, scaler = load_dataset(data, n_past, future_steps, target_col) # Load standardized training and validation

    input_size = data.shape[1]  # Adjust based on your features

    W_hidden, b_hidden, W_output, b_output = initialize_model(input_size * n_past, hidden_size, num_layers, output_size, device)

    # Combine all parameters
    params = W_hidden + b_hidden + [W_output, b_output]
    
    # Initialize the optimizer
    optimizer = torch.optim.SGD(params, lr=learning_rate)

    # Train the model
    train_model(train_loader, validation_loader, W_hidden, b_hidden, W_output, b_output, optimizer,  device)

    # Evaluate the model
    results = evaluate_model(validation_loader, W_hidden, b_hidden, W_output, b_output, num_layers, device) 

    mse_1, mae_1, mape_1, mbe_1, rmse_1, mse_3, mae_3, mape_3, mbe_3, rmse_3, mse_5, mae_5, mape_5, mbe_5, rmse_5 = results

    # record weights at this points since I can only access them here: Save improving models weights  weights of the best 
    
    avg_mse  = (mse_1+mse_3+mse_5)/3

    if avg_mse < current_best_score:
    
        current_best_score = avg_mse
        best_scaler = scaler  # Store the scaler for the best model

        best_model_weights = {
            'W_hidden': [w.detach().clone() for w in W_hidden],  # Detach and clone to avoid modifications
            'b_hidden': [b.detach().clone() for b in b_hidden],
            'W_output': W_output.detach().clone(),
            'b_output': b_output.detach().clone(),
        }
    
        # Save the best model weights and scaler
        torch.save({'model_weights': best_model_weights, 'scaler': best_scaler}, 'best_model_and_scaler.pth')
    
        print(f'run_training function MSE score: {current_best_score}')

    return results

def forward_selection(multiVarData, OG_features, n_past, future_steps, target_col, hidden_size, num_layers, output_size, learning_rate, best_score, best_model_info, device):

    current_best_features = ['Close']    # Start with the best model features as just 'Close'
    # best_score = best_model_info[3]  # Initialize with the score from the initial pass
    current_best_score = best_score
    best_scaler =  None
    best_model_weights ={}

    for feature in OG_features:

        candidate_features = current_best_features + [feature]         # Add the new feature to the current best features

        # Ensure 'Close' is always at the end of the list
        if 'Close' in candidate_features:
            candidate_features.remove('Close')
        candidate_features.append('Close')

        if 'Close' in current_best_features:
            current_best_features.remove('Close')
        current_best_features.append('Close')

        data_subset = multiVarData[candidate_features]    # Select the subset of data with the current candidate features

        train_loader, validation_loader, scaler = load_dataset(data_subset, n_past, future_steps, target_col)   # Load dataset using the modularized function (both standardized correctly)

        input_size = data_subset.shape[1]          # Initialize the model with the selected features

        W_hidden, b_hidden, W_output, b_output = initialize_model(input_size * n_past, hidden_size, num_layers, output_size, device)

        params = W_hidden + b_hidden + [W_output, b_output]         # Initialize the optimizer

        optimizer = torch.optim.SGD(params, lr=learning_rate)

        train_model(train_loader, validation_loader, W_hidden, b_hidden, W_output, b_output, optimizer, device)   # Train the model - Use early stopping to prevent overfitting

        # Evaluate the model based on feature combination - validation set
        mse_1, mae_1, mape_1, mbe_1, rmse_1, mse_3, mae_3, mape_3, mbe_3, rmse_3, mse_5, mae_5, mape_5, mbe_5, rmse_5 = evaluate_model(
            validation_loader, W_hidden, b_hidden, W_output, b_output, num_layers, device)

        # Calculate the average MSE for the 1-day, 3-day, and 5-day forecasts
        average_mse = (mse_1 + mse_3 + mse_5) / 3

        # If the new feature improves performance, keep it
        if average_mse < current_best_score:
            current_best_score = average_mse
            current_best_features.append(feature)

            best_scaler = scaler  # Store the scaler for the best model

            best_model_weights = {

                'W_hidden': [w.detach().clone() for w in W_hidden],  # Detach and clone to avoid modifications
                'b_hidden': [b.detach().clone() for b in b_hidden],
                'W_output': W_output.detach().clone(),
                'b_output': b_output.detach().clone(),
            }
    
            torch.save({'model_weights': best_model_weights, 'scaler': best_scaler}, 'best_model_and_scaler.pth')

            # Save the architecture and features of the best model (based on validation set)
            best_model_info = [n_past, hidden_size, num_layers, best_score, current_best_features.copy()]

        # Store the hyperparameters and metrics for this iteration
        hyperparams = [n_past, learning_rate, num_layers, hidden_size]
        metrics = [
            mse_1, mae_1, mape_1, mbe_1, rmse_1,
            mse_3, mae_3, mape_3, mbe_3, rmse_3,
            mse_5, mae_5, mape_5, mbe_5, rmse_5
        ]

        # Append the results to the CSV
        append_to_csv(file_name, hyperparams, metrics, candidate_features, current_best_features, feature, best_model_info)

    return current_best_score 


def record_original_performance(file_name, hyperparams, metrics, best_model_info):
    # store the results based on just the closing price
    current_best_features=['Close']
    candidate_features = ['Close']
    feature = ['Close']
    # Append the current results to the CSV, including the feature under consideration
    append_to_csv(file_name, hyperparams, metrics, candidate_features, current_best_features, feature, best_model_info)

def run_initial_pass(multiVarData, future_steps, target_col, hidden_size, num_layers, output_size, learning_rate, n_past, best_score,  device):

    best_model_info = [n_past, hidden_size, num_layers, best_score, ['Close']]  # Default, not yet optimal

    # Initial pass with just the closing price
    current_best_score = best_score
    mse_1, mae_1, mape_1, mbe_1, rmse_1, mse_3, mae_3, mape_3, mbe_3, rmse_3, mse_5, mae_5, mape_5, mbe_5, rmse_5 = run_training(
        (multiVarData['Close']).to_frame(), learning_rate, n_past, future_steps, target_col, hidden_size, num_layers, output_size, current_best_score, device
    )

    # Calculate average MSE across the 1, 3, and 5 day predictions
    avg_mse = (mse_1 + mse_3 + mse_5) / 3

    if avg_mse < current_best_score:
        # Initialize the best model information based on this initial pass
        best_model_info = [n_past, hidden_size, num_layers, avg_mse, ['Close']]
        current_best_score = avg_mse

        print(f'run_initial_pass function MSE score: {current_best_score}')


    # Store the hyperparameters and metrics for the initial pass
    hyperparams = [n_past, learning_rate, num_layers, hidden_size]
    metrics = [
        mse_1, mae_1, mape_1, mbe_1, rmse_1,
        mse_3, mae_3, mape_3, mbe_3, rmse_3,
        mse_5, mae_5, mape_5, mbe_5, rmse_5
    ]


    # Record performance of initial pass
    record_original_performance(file_name, hyperparams, metrics, best_model_info)

    return current_best_score, best_model_info


########################################### Model Data Class ##############################################


# input lagged features and labels
class multipleOutputForexDataset:
    def __init__(self, features, labels):
        self.features = []
        self.labels = []

        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)  # Change to float32
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)  # Ensure both are float32
        return features, labels


####################################### Data Loading, default values and initalisation ######################

file_path='../Data/EURUSD_D1.csv'
FXdata = data_loader(file_path)

future_steps = [1 , 3, 5]
target_col =  -1

# Initialize CSV file
file_name = 'GPU_NN_hyperparameter_search_results.csv'
initialize_csv(file_name)

#################################### Neural Network model and Architecture development ######################

# Define the hyperparameter grid
OG_features = ['Open', 'High', 'Low', 'Volume', '50_sma', '200_sma', '50_ema',
       '100_ema', 'MACD_line', 'Signal_line', 'ADX', 'RSI', 'stoch_k',
       'stoch_d', 'ATR']
output_size = 3  # Output layer size (closing price t =  [1 , 3, 5])

# hyper-parameterss
lookback_window_grid = [1, 3, 5, 7, 10]  
hidden_neurons_grid = [8 ,10, 16, 32, 64, 128]  
hidden_layers_grid = [1, 2, 3, 4]  

##############################################  Feature Engineering #########################################
multiVarData = multivariateFeatureEngineering(FXdata)
col = [col for col in multiVarData.columns if col!='Close'] + ['Close'] #Arrange the columns
multiVarData=   multiVarData[col]

###### Paramters needed for test set evaluation

best_model_info = [] # [SW, HN, HL, mse_score, features] Access afterwards for the training
best_model_features  =[] # self explanatory

# Perform training perform the split (train ,test, val)
training_set, test_valid_set =  train_test_split(multiVarData,test_size=0.2, random_state=42, shuffle=False )
validation_set, test_set  = train_test_split(test_valid_set, test_size=0.5, random_state=42, shuffle=False)

# combine taining and validation: repeat split later
training_validation_data = pd.concat([training_set, validation_set], ignore_index=False) # Pass into hyper-parametertuning

# measure grid search time
import time
start_time = time.time()
best_score = float('inf') # based on validation set

# continue the hyperparamter search on just the validation and training
for n_past in lookback_window_grid:

    for hidden_size in hidden_neurons_grid:

        for num_layers in hidden_layers_grid:

            #################### Initial Pass - Using Just Closing Price ####################
            score, best_model_info = run_initial_pass(training_validation_data, future_steps, target_col, hidden_size, num_layers, output_size, 0.001, n_past, best_score,  device)

            if score< best_score:
                best_score = score
            
            ###############################################################################################
            #                                          Peform Forward Selection                           #
            ############################################################################################### 
        
            fs_score = forward_selection(training_validation_data, OG_features, n_past, future_steps, target_col, hidden_size, num_layers, output_size, 0.001, best_score, best_model_info, device)

            if  best_model_info[3]  < best_score:
                best_score =  best_model_info[3] 
              


# ################################################# Evaluate on the test set #################################################
                
''' Using and evaluating on the test data '''

# best model get >> features, lookback window , target column>> 
test_set_best_features = test_set[best_model_features[4]]  # >> on test set, select subset of features based on best features
col = [col for col in test_set_best_features.columns if col!='Close'] + ['Close'] #Arrange the columns
test_set_best_features = test_set_best_features[col]     # >> ensure target col is at the end

# using test set with reduced features 

#load the scaler that has been saved by pytorch
saved_data = torch.load('best_model_and_scaler.pth')
scaler = saved_data['scaler'] # best models scaler

'''
NB: Ensure that use the same lag window, and features as in the best model, when creating
    lag features, to AVOID dimensionsion mismatch
'''
design_matrix, labels, _ = multivariateFeatureLagMultiStep(test_set_best_features,best_model_info[0], future_steps, target_col, scaler, False) # returned test that has been scaled b
                                                                                                                                                # based off scaler from training_and_validation
                                                                                                                                                # characteristic

# Apply the best models scaler on the test set feature
transformed_set  = multipleOutputForexDataset(design_matrix, labels)    # >> load with multipleOutputForexDataset class to generate lag features 
                                                                        # (returns standardized features as tensors)

test_set_loader = DataLoader(transformed_set, batch_size=32, shuffle=False) # >> create test dataloaders


# >> get best model architecture:
ts_input_size = test_set_best_features.shape[1]*best_model_info[0] # >> get input size = features * best_models_n_past
ts_hn = best_model_info[1]     # >> get num layers, get num_neurons, output size
ts_hl = best_model_info[2]

# Execution on the test set (use function test_set evaluations):
    # >> get best found weights: load saved weights
    # >> perform form forward pass on test set:
    # >> compute metrics on the test set
results = test_set_evaluate_model(test_set_loader, ts_hl, device )
print(f'Model performance on the test set:\n{results}')


# Record the end time
end_time = time.time()
elapsed_time = end_time - start_time # Calculate the elapsed time


print(f"Execution time for this set of parameters: {elapsed_time:.2f} seconds")
print("Grid search completed and results saved to CSV!")