import numpy as np
import os
import inspect

from tudatpy import constants
import pycode.ConjunctionUtilities as util

####################################
########## Initialisation ##########
####################################

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

# Define time interval of interest for collision analysis
tepoch = my_sat.epoch
tspan = 2 * constants.JULIAN_DAY
trange = np.array([tepoch, tepoch + tspan])


####################################             
########## Initial filter ##########
####################################

# Perigee-apogee filter
distance_pa = 10 * 10**3  # Acceptable perigee-apogee distance - m
obj_dict = util.perigee_apogee_filter(my_sat, obj_dict, distance_pa)
# 17 remaining
# print(len(obj_dict.keys()))

# Geometrical filter
distance_geom = 10 * 10**3  # Acceptable Euclidean distance - m
obj_dict, rel_distances_geom = util.geometrical_filter(my_sat, obj_dict, distance_geom)  # rel_distances_geom for debugginf purposes
# 15 remaining
# print(len(obj_dict.keys()))

# print(len(obj_dict.keys()))
# for norad_id in obj_dict.keys():
#     obj = obj_dict[norad_id]
#     print('Object perigee [km]:',obj.rp / 1000, '| Object apogee [km]:', obj.ra / 1000)


####################################
########## TCA Assessment ##########
####################################

distance_tca = 10 * 10**3  # Critical distance to identify TCAs
delete_ls = []

for norad_id in obj_dict.keys():

    obj = obj_dict[norad_id]
    # Define relevant bodies
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    # bodies = prop.tudat_initialize_bodies(bodies_to_create) 

    # Define initial cartesian states
    X1 = my_sat.cartesian_state
    X2 = obj.cartesian_state
    # Define state params
    # rso1 state params
    rso1_params = {}
    rso1_params['mass'] = my_sat.mass
    rso1_params['area'] = my_sat.area
    rso1_params['Cd'] = my_sat.Cd
    rso1_params['Cr'] = my_sat.Cr
    rso1_params['sph_deg'] = 8
    rso1_params['sph_ord'] = 8    
    rso1_params['central_bodies'] = ['Earth']
    rso1_params['bodies_to_create'] = bodies_to_create
    # rso2 state params    
    rso2_params = {}
    rso2_params['mass'] = obj.mass
    rso2_params['area'] = obj.area
    rso2_params['Cd'] = obj.Cd
    rso2_params['Cr'] = obj.Cr
    rso2_params['sph_deg'] = 8
    rso2_params['sph_ord'] = 8    
    rso2_params['central_bodies'] = ['Earth']
    rso2_params['bodies_to_create'] = bodies_to_create
    # Define integration parameters
    int_params = {}
    int_params['tudat_integrator'] = 'rkf78'
    int_params['step'] = 10.
    int_params['max_step'] = 1000.
    int_params['min_step'] = 1e-3
    int_params['rtol'] = 1e-12
    int_params['atol'] = 1e-12
    # Find TCA
    T_list, rho_list = util.compute_TCA(
        X1, X2, trange, rso1_params, rso2_params, 
        int_params, rho_min_crit = distance_tca)
    
    print('Times for TCA:',T_list,'| rho_min:', rho_list)

    # Identify possible HIEs
    if min(rho_list) < distance_tca:
        obj.tca_T_list = T_list
        obj.tca_rho_list = rho_list
    else:
        delete_ls.append(norad_id)

# Eliminate impossible impactors
for norad_id in delete_ls:
    obj_dict.pop(norad_id)
# 3 remaining
# print(len(obj_dict.keys()))


######################################
######## Propagation to TCA ##########
######################################


#######################################
######## Detailed assessment ##########
#######################################


########################################
######## Manoeuvre assessment ##########
########################################  



