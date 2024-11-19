#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
concat = __import__('11-concat').concat

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df = concat(df1, df2)

print(df)
