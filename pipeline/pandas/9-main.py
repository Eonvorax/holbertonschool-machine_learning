#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
fill = __import__('9-fill').fill

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = fill(df)

print(df.head())
print(df.tail())
