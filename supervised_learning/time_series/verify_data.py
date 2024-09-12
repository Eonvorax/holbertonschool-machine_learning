#!/usr/bin/env python3

import numpy as np

# Check the shape of the data in the npz file
data = np.load("preprocessed_data_raw.npz")
# Should print something like (samples, timesteps, features)
print(data['data'].shape)
# Should print something like (samples, timesteps)
print(data['targets'].shape)
