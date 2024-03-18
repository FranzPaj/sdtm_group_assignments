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
obj_dict = util.read_catalog_file(fname)
num_objs = len(obj_dict.keys())

# Define our object
my_norad = 40940
my_sat = util.Object(obj_dict, my_norad)
obj_dict.pop(my_norad)

# Define a list of relevant possible impactors as class objects
obj_ls = list()  # Object list
for elem in obj_dict.keys():
    # Each element of the objects list is an object with the relevant properties
    Obj = util.Object(obj_dict, elem)
    obj_ls.append(Obj)





########## Initial filter ##########
    
########## TCA Assessment ##########

######## Detailed assessment ##########
    
######## cManoeuvre assessment ##########



