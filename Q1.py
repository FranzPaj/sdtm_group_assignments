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

# Perigee-apogee filter
distance_pa = 10 * 10**3  # Acceptable perigee-apogee distance - m
obj_dict = util.perigee_apogee_filter(my_sat, obj_dict, distance_pa)
# 17 remaining
print(len(obj_dict.keys()))

# Geometrical filter
distance_geom = 10 * 10**3  # Acceptable Euclidean distance - m
obj_dict = util.geometrical_filter(my_sat, obj_dict, distance_geom)
# 15 remaining
# print(len(obj_dict.keys()))
for norad_id in obj_dict.keys():
    obj = obj_dict[norad_id]
    print('Object perigee [km]:',obj.rp / 1000, '| Object apogee [km]:', obj.ra / 1000)

########## TCA Assessment ##########

######## Detailed assessment ##########
    
######## cManoeuvre assessment ##########



