import numpy as np
import os
import inspect

from tudatpy.astro.time_conversion import epoch_from_date_time_iso_string, epoch_from_date_time_components, datetime_to_tudat

import pycode.ConjunctionUtilities as util

class Object:

    def __init__(self,obj_dict,elem):

        # Define the orbital elements and characteristic of a certain spacecraft or debris
        self.NORAD_ID = elem  # NORAD ID of the element
        self.utc = obj_dict[elem]['UTC']  # UTC of the time corresponding to the object state
        self.epoch = epoch_from_date_time_components(
            self.utc.year, self.utc.month, self.utc.day, self.utc.hour, 
            self.utc.minute, float(self.utc.second))  # epoch since J2000 for TudatPy - s
        self.state = obj_dict[elem]['state']  # Cartesian state at epoch
        self.covar = obj_dict[elem]['covar']  # Uncertainty covariance at epoch
        self.mass = obj_dict[elem]['mass']  # Object mass
        self.area = obj_dict[elem]['area']  # Object reference area
        self.Cd = obj_dict[elem]['Cd']  # Object drag coefficient
        self.Cr = obj_dict[elem]['Cr']  # Object radiation pressure coefficient
        

# Define path to objects datafile
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
data_dir = os.path.join(current_dir,'data','group2')
fname = os.path.join(data_dir, 'estimated_rso_catalog.pkl')
# Read the relevant objects datafile
obj_dict = util.read_catalog_file(fname)
num_objs = len(obj_dict.keys())

# Define a list of relevant objects as class objects
objs_ls = list()  # Objects list
for elem in obj_dict.keys():
    # Each element of the objects list is an object with the relevant properties
    Obj = Object(obj_dict, elem)
    objs_ls.append(Obj)

