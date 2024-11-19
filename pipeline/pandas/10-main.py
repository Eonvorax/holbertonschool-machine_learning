#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
index = __import__('10-index').index

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = index(df)

print(df.tail())
