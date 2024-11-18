#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
array = __import__('4-array').array

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

A = array(df)

print(A)
