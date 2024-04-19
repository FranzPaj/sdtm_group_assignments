import numpy as np
import os
import inspect
from math import ceil, log
from pycode.CDM import write_cdm
import pickle

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
distance_pa = 25 * 10 ** 3  # Acceptable perigee-apogee distance - m
obj_dict = util.perigee_apogee_filter(my_sat, obj_dict, distance_pa)
# 17 remaining
print('Perigee-apogee filter:', len(obj_dict.keys()), 'remaining')

# Geometrical filter
distance_geom = 25 * 10 ** 3  # Acceptable Euclidean distance - m
obj_dict_, rel_distances_geom = util.geometrical_filter(my_sat, obj_dict,
                                                        distance_geom)  # rel_distances_geom for debugginf purposes
# 17 remaining
print('Geometrical filter:', len(obj_dict.keys()), 'remaining')

# Time filter
distance_time = 25 * 10 ** 3
obj_dict_2 = obj_dict.copy()
obj_dict_2 = util.time_filter(my_sat, obj_dict, tspan, distance_time)
print('Temporal filter:', len(obj_dict.keys()), 'remaining')

# print(len(obj_dict.keys()))
for norad_id in obj_dict.keys():
    obj = obj_dict[norad_id]
    print('Object perigee [km]:', obj.rp / 1000, '| Object apogee [km]:', obj.ra / 1000)

####################################
########## TCA Assessment ##########
####################################

print('-------------------------------')
print('TCA Assessment')
print('-------------------------------')

distance_tca = 20 * 10 ** 3  # Critical distance to identify TCAs
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
######## Finding best TCA  ###########
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

int_params_propagation_to_epoch = {'tudat_integrator': 'rk4', 'step': 1}

int_params_precise_analysis_forward = {'tudat_integrator': 'rkf78', 'step': 0.01, 'max_step': 0.05, 'min_step': 1e-4,
                                       'rtol': 1e-12, 'atol': 1e-12}

int_params_precise_analysis_backwards = {'tudat_integrator': 'rkf78', 'step': -0.01, 'max_step': -0.05,
                                         'min_step': -1e-4, 'rtol': 1e-12, 'atol': 1e-12}

for norad_id in obj_dict.keys():
    obj = obj_dict[norad_id]
    Xo = obj.cartesian_state
    rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
                  'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}

    print('RSO :', norad_id)
    print('Propagation around estimated TCA to find better approximation')
    # First propagation to final epoch
    tvec = np.array([obj.epoch, obj.tca_T_list[0]])
    t_rso, X_rso_prop = propagate_orbit(Xo, tvec, rso_params, int_params_propagation_to_epoch)
    print('RSO normal prop done')
    t_sat, X_sat_prop = propagate_orbit(Xo_sat, tvec, sat_params, int_params_propagation_to_epoch)
    print('SAT normal prop done')

    # Propagation around epoch
    end_time = obj.tca_T_list[0] - 10
    tvec_backwards = np.array([obj.tca_T_list[0], end_time])
    t_rso_back, X_rso_back = propagate_orbit(X_rso_prop[-1, :], tvec_backwards, rso_params,
                                             int_params_precise_analysis_backwards)
    print('RSO backwards prop done')
    t_sat_back, X_sat_back = propagate_orbit(X_sat_prop[-1, :], tvec_backwards, sat_params,
                                             int_params_precise_analysis_backwards)
    print('SAT backwards prop done')
    tvec_forward = np.array([obj.tca_T_list[0], obj.tca_T_list[0] + 10])
    t_rso_for, X_rso_for = propagate_orbit(X_rso_prop[-1, :], tvec_forward, rso_params,
                                           int_params_precise_analysis_forward)
    print('RSO forward prop done')
    t_sat_for, X_sat_for = propagate_orbit(X_sat_prop[-1, :], tvec_forward, sat_params,
                                           int_params_precise_analysis_forward)
    print('SAT backwards prop done')

    check = False
    tca_real = obj.tca_T_list[0]
    min_dist = obj.tca_rho_list[0]
    for i in range(len(t_rso_back)):
        dist = np.linalg.norm(X_rso_back[i, 0:3] - X_sat_back[i, 0:3])
        if dist < min_dist:
            check = True
            min_dist = dist
            tca_real = t_rso_back[i]

    for i in range(len(t_rso_for)):
        dist = np.linalg.norm(X_rso_for[i, 0:3] - X_sat_for[i, 0:3])
        if dist < min_dist:
            check = True
            min_dist = dist
            tca_real = t_rso_for[i]

    if check is True:
        print('Final closest distance: ', min_dist, '\n')
        print('Epoch: ', tca_real, '\n')
        print('Difference from Denenberg method:\n'
              'Dist difference: ', min_dist - obj.tca_rho_list[0], '\n'
              'Time diff: ', tca_real - obj.tca_T_list[0])
    else:
        print('Final closest distance and TCA coincident with Denenberg method\n'
              'Min distance: ', min_dist, '\n'
              'TCA: ', obj.tca_T_list[0])
    print('-------------------------------\n')
    obj.tca_T_list[0] = tca_real
    obj.tca_rho_list[0] = min_dist



######################################
######## Propagation to TCA ##########
######################################
# intergator parameters
int_params = {'tudat_integrator': 'rkf78', 'step': 10., 'max_step': 1000., 'min_step': 1e-3, 'rtol': 1e-12,
              'atol': 1e-12}

for norad_id in obj_dict.keys():

    obj = obj_dict[norad_id]
    Xo = obj.cartesian_state
    Po = obj.covar
    print(np.shape(Xo))

    rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
                  'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}

    tf = np.zeros(len(obj.tca_T_list))
    Xf = np.zeros(shape=(6, len(obj.tca_T_list)))
    Pf = np.zeros(shape=(6, 6 * len(obj.tca_T_list)))

    prop = {'tf': np.array([]), 'Xf': np.array([]), 'Pf': np.array([])}

    for i in range(len(obj.tca_T_list)):
        tvec = np.array([obj.epoch, obj.tca_T_list[i]])
        tf[i], Xf_new, Pf_matrix = propagate_state_and_covar(Xo, Po, tvec, rso_params, int_params)
        Xf[:, i] = Xf_new.reshape((6,))
        inx = 6 * i
        Pf[:, inx:(inx + 6)] = Pf_matrix

        tvec_sat = np.array([my_sat.epoch, obj.tca_T_list[i]])
        tf_sat, Xf_sat, Pf_matrix_sat = propagate_state_and_covar(Xo_sat, Po_sat, tvec_sat, sat_params, int_params)

        if len(prop['tf']) == 0:
            prop['tf'] = tf_sat
        else:
            prop['tf'] = [prop['tf'], tf_sat]

        if len(prop['Xf']) == 0:
            prop['Xf'] = Xf_sat
        else:
            prop['Xf'] = np.column_stack((prop['Xf'], Xf_sat))

        if len(prop['Pf']) == 0:
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

    HBR = np.sqrt(my_sat.area / np.pi) + np.sqrt(obj.area / np.pi)
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


    filename = write_cdm(obj.utc, obj.tca_T_list[0], d, M, np.linalg.norm(rel_vel), rel_pos_ric,
                         Pc, norad_id, Xf_sat, Xf_rso, Pf_sat, Pf_rso)

    print('CDM generated and saved in: ', filename)
    print('\n-------------------------------')
    print('-------------------------------')

########################################
######## Data for NASA MonteCarlo ######
########################################

montecarlo_data_dir = os.path.join(current_dir, 'CARA_matlab', 'MonteCarloPc', 'data_files')
obj_filename = os.path.join(montecarlo_data_dir, 'obj_dict_test.txt')

matlab_data = np.zeros((len(obj_dict.keys()), 88))
i = 0
# Change into comprehensible format for matlab:
for norad_id in obj_dict.keys():
    obj = obj_dict[norad_id]
    matlab_data[i, 0] = norad_id
    matlab_data[i, 1] = obj.area
    matlab_data[i, 2:8] = obj.Xf[:, 0]
    matlab_data[i, 8:44] = obj.Pf.flatten()

    norad_id_str = str(norad_id)
    Xf_sat = getattr(my_sat, norad_id_str)['Xf']
    Pf_sat = getattr(my_sat, norad_id_str)['Pf']
    matlab_data[i, 44] = my_sat.NORAD_ID
    matlab_data[i, 45] = my_sat.area
    matlab_data[i, 46:52] = Xf_sat[:, 0]
    matlab_data[i, 52:88] = Pf_sat.flatten()

    i += 1

np.savetxt(obj_filename, matlab_data)
