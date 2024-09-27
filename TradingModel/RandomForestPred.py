import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Import dataset
df = pd.read_csv('./Data/EURUSD_D1.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

print(df.info())

# Feature and Target Extraction
X = df[['Open', 'High', 'Low']]
Y = df['Close']

print(X.head(), Y.head())

# Test and training split (80/20)
shape_80 = int(X.shape[0] * 0.8)

x_train = X[:shape_80]
y_train = Y[:shape_80]

x_test = X[shape_80:]
y_test = Y[shape_80:]
test_dates = df['Time'][shape_80:]

# Model training and fit
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
regressor.fit(x_train, y_train)

# Model Predictions and Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# OOB score
oob_score = regressor.oob_score_
print(f'OOB Score: {oob_score}')

# Prediction on test
predictions = regressor.predict(x_test)
predictions = pd.DataFrame(predictions, columns=['Predicted Price'])

mse = mean_squared_error(y_test, predictions)
print(f'Mean squared error: {mse}')

# Plotting
predictions.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

combined = pd.concat([test_dates, predictions], axis=1)

y_test = y_test.reset_index(drop=True)
y_test.name = 'Close'
testData_combined = pd.concat([test_dates, y_test], axis=1)

plt.plot(testData_combined['Time'], testData_combined['Close'], color='b', label='True Price')
plt.plot(combined['Time'], combined['Predicted Price'], color='r', label='Predicted')

plt.xlabel('Dates')
plt.ylabel('Price')
plt.title("True and Predicted Test Set Results")
plt.legend()

plt.show()



# Trade simulation ideas #

'''
Dataframe to keep track of trades
'''
#  # Initialize lists to store trade data
# trades = []

# # Generate signals and simulate trades
# for i in range(len(y_pred_test) - 1):  # Adjust the loop to stop before the last index
#     entry_time = time_test.values[i]
#     entry_price = y_test.values[i]

#     if y_pred_test[i] > X_test[i][0]:  # Predicted close > open (long position signal)
#         stop_loss = entry_price - 0.0010  # Assuming 10 pips for long
#         take_profit = entry_price + 0.0030  # Assuming 30 pips for long

#         # Simulate trade
#         if y_test.values[i + 1] <= stop_loss:
#             pips = (stop_loss - entry_price) * 10000  # Stopped out
#         elif y_test.values[i + 1] >= take_profit:
#             pips = (take_profit - entry_price) * 10000  # Take profit hit
#         else:
#             pips = (y_test.values[i + 1] - entry_price) * 10000  # Closed at next candle

#         trades.append([entry_time, entry_price, stop_loss, take_profit, pips])

#     elif y_pred_test[i] < X_test[i][0]:  # Predicted close < open (short position signal)
#         stop_loss = entry_price + 0.0010  # Adjust for short position
#         take_profit = entry_price - 0.0030  # Adjust for short position

#         # Simulate trade for short
#         if y_test.values[i + 1] >= stop_loss:
#             pips = (entry_price - stop_loss) * 10000  # Stopped out
#         elif y_test.values[i + 1] <= take_profit:
#             pips = (entry_price - take_profit) * 10000  # Take profit hit
#         else:
#             pips = (entry_price - y_test.values[i + 1]) * 10000  # Closed at next candle

#         trades.append([entry_time, entry_price, stop_loss, take_profit, pips])

#         '''
#         cant I make this continue to the next iteration if the until eiher the stop loss of the take profit is hit? 
#         That would be a proper simulation of a trade
#         '''

# # Create DataFrame from trades
# trades_df = pd.DataFrame(trades, columns=['Entry Time', 'Entry Price', 'Stop Loss', 'Take Profit', 'Pips Gained/Lost'])

# # Export DataFrame to a CSV file
# trades_df.to_csv('trades_results.csv', index=False)  # 'index=False' avoids writing row numbers

# print(trades_df.head())

'''
Section End
'''