#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df: pd.DataFrame = from_file(
    'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted price column
df = df.drop(columns=["Weighted_Price"])

# Rename the 'Timestamp' column to 'Datetime' using mapping dict
df = df.rename(columns={"Timestamp": "Date"})

# Convert the renamed column to the datetime format
df["Date"] = pd.to_datetime(df["Date"], unit='s')

# Index the DataFrame on Date column
df = df.set_index(["Date"])

# Set missing values in Close to previous row value
df["Close"] = df["Close"].ffill()

# Fill missing values in these columns with corresponding "Close" values
for column in ["High", "Low", "Open"]:
    if column in df.columns:
        df[column] = df[column].fillna(df["Close"])

# Set missing values in the Volume columns to 0
for column in ["Volume_(BTC)", "Volume_(Currency)"]:
    if column in df.columns:
        df[column] = df[column].fillna(0)

# Filter data from 2017 onward
df = df[df.index >= "2017-01-01"]


# Resample and aggregate data at daily intervals
daily_df = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

print(daily_df)

# Plotting the data
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot High, Low, Open, Close
axs[0].plot(daily_df.index, daily_df["High"], label="High", color="green")
axs[0].plot(daily_df.index, daily_df["Low"], label="Low", color="red")
axs[0].plot(daily_df.index, daily_df["Open"], label="Open", color="orange")
axs[0].plot(daily_df.index, daily_df["Close"], label="Close", color="blue")
axs[0].set_title("Daily Prices (High, Low, Open, Close)", fontsize=16)
axs[0].set_ylabel("Price (USD)", fontsize=14)
axs[0].legend()
axs[0].grid()

# Plot Volume_(BTC)
axs[1].plot(daily_df.index, daily_df["Volume_(BTC)"],
            label="Volume (BTC)", color="purple")
axs[1].set_title("Daily Volume (BTC)", fontsize=16)
axs[1].set_ylabel("Volume (BTC)", fontsize=14)
axs[1].legend()
axs[1].grid()

# Plot Volume_(Currency)
axs[2].plot(daily_df.index, daily_df["Volume_(Currency)"],
            label="Volume (Currency)", color="brown")
axs[2].set_title("Daily Volume (Currency)", fontsize=16)
axs[2].set_ylabel("Volume (USD)", fontsize=14)
axs[2].set_xlabel("Date", fontsize=14)
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.savefig("Visualize_BTC_2017-2019.png")
plt.show()
