#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
analyze = __import__('13-analyze').analyze

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

stats = analyze(df)

print(stats)
