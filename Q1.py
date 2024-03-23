import numpy as np
import os
import inspect
from math import ceil, log
from pycode.CDM import write_cdm

from tudatpy import constants
import pycode.ConjunctionUtilities as util
from pycode.TudatPropagator import *

####################################
########## Initialisation ##########
####################################

# Define path to objects datafile
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
data_dir = os.path.join(current_dir, 'data', 'group2')
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

print(np.rad2deg(my_sat.keplerian_state[1]))

print('Initial list:', len(obj_dict.keys()), 'objects')

# Perigee-apogee filter
distance_pa = 10 * 10 ** 3  # Acceptable perigee-apogee distance - m
obj_dict = util.perigee_apogee_filter(my_sat, obj_dict, distance_pa)
# 17 remaining
print('Perigee-apogee filter:', len(obj_dict.keys()), 'remaining')

# Geometrical filter
distance_geom = 10 * 10 ** 3  # Acceptable Euclidean distance - m
obj_dict_, rel_distances_geom = util.geometrical_filter(my_sat, obj_dict,
                                                        distance_geom)  # rel_distances_geom for debugginf purposes
# 17 remaining
print('Geometrical filter:', len(obj_dict.keys()), 'remaining')

# Time filter
distance_time = 10 * 10 ** 3
obj_dict_2 = obj_dict.copy()
obj_dict_2 = util.time_filter(my_sat, obj_dict, tspan, distance_time)
print('Temporal filter:', len(obj_dict.keys()), 'remaining')

# # print(len(obj_dict.keys()))
# for norad_id in obj_dict.keys():
#     obj = obj_dict[norad_id]
#     print('Object perigee [km]:',obj.rp / 1000, '| Object apogee [km]:', obj.ra / 1000)


####################################
########## TCA Assessment ##########
####################################

print('-------------------------------')
print('TCA Assessment')
print('-------------------------------')

distance_tca = 10 * 10 ** 3  # Critical distance to identify TCAs
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
        int_params, rho_min_crit=distance_tca)

    print('NORAD ID:', norad_id)
    print('Times for TCA [hr since epoch]:', (np.array(T_list) - tepoch) / constants.JULIAN_DAY * 24, '| rho_min [m]:',
          rho_list)
    print('-------------------------------')

    # Identify possible HIEs
    if min(rho_list) < distance_tca:
        obj.tca_T_list = T_list
        obj.tca_rho_list = rho_list
    else:
        delete_ls.append(norad_id)

# Eliminate impossible impactors
for norad_id in delete_ls:
    obj_dict.pop(norad_id)
# 5 remaining
print('TCA cutoff:', len(obj_dict.keys()), 'remaining')

######################################
######## Propagation to TCA ##########
######################################

bodies_to_create = ['Sun', 'Earth', 'Moon']

sat_params = {}
sat_params['mass'] = my_sat.mass
sat_params['area'] = my_sat.area
sat_params['Cd'] = my_sat.Cd
sat_params['Cr'] = my_sat.Cr
sat_params['sph_deg'] = 8
sat_params['sph_ord'] = 8
sat_params['central_bodies'] = ['Earth']
sat_params['bodies_to_create'] = bodies_to_create
Xo_sat = my_sat.cartesian_state
Po_sat = my_sat.covar

# intergator parameters
int_params = {'tudat_integrator': 'rkf78', 'step': 10., 'max_step': 1000., 'min_step': 1e-3, 'rtol': 1e-12,
              'atol': 1e-12}

for norad_id in obj_dict.keys():

    obj = obj_dict[norad_id]
    Xo = obj.cartesian_state
    Po = obj.covar

    rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
                  'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}

    tf = np.zeros(len(obj.tca_T_list))
    Xf = np.zeros(shape=(6, len(obj.tca_T_list)))
    Pf = np.zeros(shape=(6, 6*len(obj.tca_T_list)))

    prop = {'tf': np.array([]), 'Xf': np.array([]), 'Pf': np.array([])}

    for i in range(len(obj.tca_T_list)):
        tvec = np.array([obj.epoch, obj.tca_T_list[i]])
        tf[i], Xf_new, Pf_matrix = propagate_state_and_covar(Xo, Po, tvec, rso_params, int_params)
        Xf[:, i] = Xf_new.reshape((6, ))
        inx = 6*i
        Pf[:, inx:(inx+6)] = Pf_matrix

        tvec_sat = np.array([my_sat.epoch, obj.tca_T_list[i]])
        tf_sat, Xf_sat, Pf_matrix_sat = propagate_state_and_covar(Xo_sat, Po_sat, tvec_sat, sat_params, int_params)

        if prop['tf'].size == 0:
            prop['tf'] = tf_sat
        else:
            prop['tf'] = [prop['tf'], tf_sat]

        if prop['Xf'].size == 0:
            prop['Xf'] = Xf_sat
        else:
            prop['Xf'] = np.column_stack((prop['Xf'], Xf_sat))

        if prop['Pf'].size == 0:
            prop['Pf'] = Pf_matrix_sat
        else:
            prop['Pf'] = np.column_stack((prop['Pf'], Pf_matrix_sat))

    # print(my_sat.norad_id)
    obj.tf = tf
    obj.Xf = Xf
    obj.Pf = Pf

    setattr(my_sat, str(norad_id), prop)

#######################################
######## Detailed assessment ##########
#######################################
print('-------------------------------')
print('Detailed assessment')
print('-------------------------------')

for norad_id in obj_dict.keys():
    obj = obj_dict[norad_id]
    norad_id_str = str(norad_id)
    print('RSO :', norad_id_str)

    Xf_sat = getattr(my_sat, norad_id_str)['Xf']
    Pf_sat = getattr(my_sat, norad_id_str)['Pf']
    # print(Pf_sat[0:3, 0:3])

    Xf_rso = obj.Xf
    Pf_rso = obj.Pf

    HBR = np.sqrt(my_sat.area/np.pi) + np.sqrt(obj.area/np.pi)
    Pc = util.Pc2D_Foster(Xf_sat, Pf_sat, Xf_rso, Pf_rso, HBR, rtol=1e-8, HBR_type='circle')

    M = util.compute_mahalanobis_distance(Xf_sat[0:3], Xf_rso[0:3], Pf_sat[0:3, 0:3], Pf_rso[0:3, 0:3])
    d = util.compute_euclidean_distance(Xf_sat[0:3], Xf_rso[0:3])

    print('Mahalanobis distance:\t\t\t\t ', M)
    print('Foster Probability of collision:\t ', Pc)
    # print('Euclidean distance:\t ', d)

    ''' Broken for now (Pc not psd)
    if Pc > 1e-6:
        N_Mc = 10**(ceil(log(1/Pc * 10, 10)))
        print('Number of points for Montecarlo analysis:\t', N_Mc)

        P_mc = util.montecarlo_Pc_final(Xf_sat[0:3], Xf_rso[0:3], Pf_sat[0:3, 0:3], Pf_rso[0:3, 0:3], HBR, N=N_Mc)
        print('Montecarlo Probability of collision:\t', P_mc)
    '''

    Xf_sat_ric = util.eci2ric(Xf_sat[0:3], Xf_sat[3:6], Xf_sat[0:3])
    Xf_obj_ric = util.eci2ric(Xf_sat[0:3], Xf_sat[3:6], Xf_rso[0:3])
    print('Radial distance:\t\t\t\t ', np.abs(Xf_sat_ric[0] - Xf_obj_ric[0])[0])
    print('Along Track distance:\t\t\t ', np.abs(Xf_obj_ric[1])[0])
    print('Cross Track distance:\t\t\t ', np.abs(Xf_obj_ric[2])[0])

    print('USEFUL INFORMATION')
    rel_pos = Xf_rso[0:3] - Xf_sat[0:3]
    rel_vel = Xf_rso[3:6] - Xf_sat[3:6]
    rel_pos_ric = util.eci2ric(Xf_sat[0:3], Xf_sat[3:6], rel_pos)
    rel_vel_ric = util.eci2ric_vel(Xf_sat[0:3], Xf_sat[3:6], rel_pos_ric, rel_vel)
    print('Relative Velocity in ECI:\n', rel_vel)
    print('Relative Velocity in RSW:\n', rel_vel_ric)

    # write_cdm(epoch, tca, d_euc, speed, rel_pos_ric, Pc, norad_rso, x_sat, x_rso, P_sat, P_rso)
    filename = write_cdm(obj.utc, obj.tca_T_list[0], d, np.linalg.norm(rel_vel), rel_pos_ric,
                         Pc, norad_id, Xf_sat, Xf_rso, Pf_sat, Pf_rso)

    print('CDM generated and saved in: ', filename)
    print('\n-------------------------------')








