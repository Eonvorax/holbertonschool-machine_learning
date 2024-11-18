#!/usr/bin/env python3
"""
Dataframe from a dictionary
"""
import pandas as pd


df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]
    },
    index=list("ABCD")
)
