import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data and pre-process
df = pd.read_csv('./gcp_billing_data_20240816 - gcp_billing_data_20240816(in).csv')
df['usage_end_time'] = pd.to_datetime(df['usage_end_time'], format='%m/%d/%Y %H:%M')

print(df.head())

chunk0 =  df[:20000]
print(f"shape of chunk: {chunk0.shape} \nand information on data: \n{chunk0.head()}")


plt.figure(figsize=(12,6))
plt.plot(chunk0['usage_end_time'], chunk0['cost'])
plt.plot(df['usage_end_time'], df['cost'])
plt.xlabel('Dates')
plt.ylabel('cost')
plt.show()

# sns.set()
sns.set_theme(style="whitegrid")
sns.lineplot(data=chunk0, x='usage_end_time', y='cost', marker='o')  # Add markers if desired
plt.title('Chunk 0')
plt.show()