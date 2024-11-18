#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
rename = __import__('3-rename').rename

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = rename(df)

print(df.tail())
