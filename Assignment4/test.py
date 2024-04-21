import numpy as np
import pickle
import os
import inspect

# x = np.arange(10)
# print(x)
# print(x[-10:-1])

# Define path to objects datafile
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
data_dir = os.path.join(current_dir, 'data', 'group2')
fname = os.path.join(data_dir, 'estimated_rso_catalog.pkl')

filename = 'rso_estimated.pkl'
file_path = os.path.join(current_dir,'Q3_plots',filename)
with open(file_path, 'rb') as f:
    filter_iod_dict = pickle.load(f)

print(filter_iod_dict[98765].keys())

print(filter_iod_dict[98765]['covar'])
print(filter_iod_dict[98765]['Cr'])
