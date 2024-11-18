#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
flip_switch = __import__('6-flip_switch').flip_switch

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = flip_switch(df)

print(df.tail(8))
