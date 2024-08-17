# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adtk.detector import ThresholdAD
from adtk.visualization import plot
from adtk.data import validate_series

# Load data and set usage_end_time as index with correct datetime parsing
df = pd.read_csv('~/Desktop/myProjects/Machine-Learning-and-Python-Projects/Hackathon/gcp_billing_data_20240816 - gcp_billing_data_20240816(in).csv', 
                 index_col="usage_end_time", 
                 parse_dates=["usage_end_time"])

# Get the unique service types
unique_service_types = df['service_type'].unique()
print(f"Unique Service Types: \n{unique_service_types}\n ")

# Filter and analyze a specific service type (e.g., 'Cloud DNS')
df_service_type = df[df['service_type'] == 'Cloud DNS']

# Display the first few rows
print(df_service_type.head())

# Validate the 'cost' series for anomaly detection
s = validate_series(df_service_type['cost'])

# Apply ThresholdAD for anomaly detection
threshold_ad = ThresholdAD(high=2, low=0)
anomalies = threshold_ad.detect(s)

plt.style.use("seaborn-v0_8-whitegrid")

# Plot the data with anomalies
plot(s, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

# Display the plot
plt.show()
