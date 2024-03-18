import numpy as np
import os
import inspect

import pycode.ConjunctionUtilities as util

        
########## Initialisation ##########

# Define path to objects datafile
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
data_dir = os.path.join(current_dir,'data','group2')
fname = os.path.join(data_dir, 'estimated_rso_catalog.pkl')
# Read the relevant objects datafile
data_dict = util.read_catalog_file(fname)
num_objs = len(data_dict.keys())

# Define our object
my_norad = 40940
my_sat = util.Object(data_dict, my_norad)
data_dict.pop(my_norad)

# Define a dictionary of relevant possible impactors as class objects
obj_dict = dict()  # Object dictionary
for elem in data_dict.keys():
    # Each element of the possible impactors dictionary is an object with useful properties
    Obj = util.Object(data_dict, elem)
    obj_dict[elem] = Obj
             
########## Initial filter ##########

distance_pa = 3 * 10**3  # Acceptable perigee-apogee distance
obj_ls = util.perigee_apogee_filter(my_sat, obj_dict, distance_pa)
    
########## TCA Assessment ##########

######## Detailed assessment ##########
    
######## cManoeuvre assessment ##########



