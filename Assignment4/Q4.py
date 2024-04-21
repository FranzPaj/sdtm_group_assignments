#### Code for Question 4 ####

import numpy as np
import os
import inspect
from math import ceil, log
from pycode.CDM import write_cdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tudatpy.astro.time_conversion import date_time_from_epoch, datetime_to_python
from tudatpy.astro.element_conversion import cartesian_to_keplerian

from tudatpy import constants
import pycode.ConjunctionUtilities as util
from pycode.ConjunctionUtilities import mu_e
import pycode.EstimationUtilities as estimation
from pycode.TudatPropagator import *
from pycode import plot_gen

####################################
########## Initialisation ##########
####################################

# Define path to objects datafile
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
data_dir = os.path.join(current_dir, 'data', 'group2')
fname = os.path.join(data_dir, 'estimated_rso_catalog.pkl')
# Read the relevant objects datafile
data_dict = util.read_catalog_file(fname)
# Add the IOD object
iod_norad_id = 99002
filename = 'rso_estimated.pkl'
file_path = os.path.join(current_dir,'output','estimations', 'q3',filename)
with open(file_path, 'rb') as f:
    filter_iod_dict = pickle.load(f)
data_dict[iod_norad_id] = filter_iod_dict[98765]

num_objs = len(data_dict.keys())


# Define a dictionary of relevant possible impactors as class objects
### 91686 is the manoeuvre one!
### 99002 is the initial orbit determination
### 91762 is the object characterisation
close_encounter_ls = [91883, 40940]
# close_encounter_ls = [91332, 91395, 91509, 91686, 91883, 40940]
new_encounter_ls = [91762]

#### Is 99002 a new one? What is that? Check

obj_dict = dict()  # Object dictionary
new_obj_dict = dict()
for elem in data_dict.keys():
    if elem in close_encounter_ls:
        # Each element of the possible impactors dictionary is an object with useful properties
        Obj = util.Object(data_dict, elem)
        obj_dict[elem] = Obj
    elif elem in new_encounter_ls:
        Obj = util.Object(data_dict, elem)
        new_obj_dict[elem] = Obj
    elif elem == 99002:
        Obj = util.Object(data_dict, elem)
        obj_dict[elem] = Obj
# Add our object
my_norad = 40940
my_sat = util.Object(data_dict, my_norad)
obj_dict[my_norad] = my_sat  # ADD LATER, COMMENTED ONLY FOR DEBUGGING

# rso_91883_nom = obj_dict[]

# Define time interval of interest for collision analysis
tepoch = my_sat.epoch
tspan = 2 * constants.JULIAN_DAY
tend = tepoch + tspan
trange = np.array([tepoch, tend])


################################################
########## Measurement processing ##############
################################################

print('-------------------------------')
print('Processing of new data and measurements')
print('-------------------------------')
print()

# Define parameters for filter and propagation
int_params = {'tudat_integrator': 'rkf78',
              'step': 10.,
              'max_step': 1000.,
              'min_step': 1e-3,
              'rtol': 1e-12,
              'atol': 1e-12}
# Define boundaries on residuals to consider, to avoid initial imprecise estimations
boundary_dict = {
    40940: 0,
    91332: 0,
    91395: 0,
    91509: 0,
    91686: 12,
    91762: 0,
    91883: 0,
    99002: 0,
}

# Extract estimation from measurements using a UKF
for norad_id in close_encounter_ls:

    rso = obj_dict[norad_id]

    # Housekeeping messages to terminal
    print('RSO:', norad_id)
    print('-------------UKF started-------------')

    state_params, meas_dict, sensor_params = estimation.read_measurement_file(
        os.path.join(current_dir, 'data', 'states_updated', 'q4_meas_rso_'+str(norad_id)+'.pkl'))

    # Observed parameters: - ra
    #                      - dec
    # Noise of the sensor: - 'ra': 4.84813681109536e-06 rad, 1 arcsec
    #                      - 'dec': 4.84813681109536e-06 rad, 1 arcsec

    
    # Consider RSO which executed manoeuvre
    if norad_id == 91686:
        # Elaborate maneouvre estimation data
        filename = 'ukf_afterman.pkl'
        file_path = os.path.join(current_dir,'output', 'estimations', 'q2',filename)
        with open(file_path, 'rb') as f:
            filter_man_dict = pickle.load(f)
        final_time_man = list(filter_man_dict)[-1]
        final_state_man = filter_man_dict[final_time_man]['state']
        final_covar_man = filter_man_dict[final_time_man]['covar']
        final_covar_man = util.remediate_covariance(final_covar_man,1.0E-09)[0]
        state_params['state'] = final_state_man
        state_params['covar'] = final_covar_man
        state_params['UTC'] = datetime_to_python(date_time_from_epoch(final_time_man))
        # print(state_params['UTC'])
        # print(datetime_to_python(date_time_from_epoch(final_time_man)))
        # Eliminate the pre-manoeuvre data
        time_manoeuvre = tepoch +  8 * 3600  # s
        time_meas = meas_dict['tk_list']
        # print((time_meas[0] - tepoch)/3600)
        ind_manoeuvre = np.digitize(time_manoeuvre, time_meas) + 1
        meas_dict['tk_list'] = meas_dict['tk_list'][ind_manoeuvre:]
        meas_dict['Yk_list'] = meas_dict['Yk_list'][ind_manoeuvre:]

    # Optical parameters
    # Setup 1
    # filter_params = {
    #     'Qeci': 1.0E-09 * np.diag([1., 1., 1.]),
    #     'Qric': 0 * np.diag([1., 1., 1.]),
    #     'alpha': 1.0E-4,
    #     'gap_seconds': 100
    # }
    # Setup 2
    filter_params = {
        'Qeci': 0 * np.diag([1., 1., 1.]),
        'Qric': 5.0E-14 * np.diag([1., 1., 1.]),
        'alpha': 1.0E-03,
        'gap_seconds': 100
    }

    # if norad_id == 91883:
    #     time_split = tepoch + 5 * 3600
    #     time_new_meas = meas_dict['tk_list']
    #     ind_new_meas = np.digitize(time_split,time_new_meas) + 1
    #     meas_dict['tk_list'] = meas_dict['tk_list'][ind_new_meas:]
    #     meas_dict['Yk_list'] = meas_dict['Yk_list'][ind_new_meas:]

    filter_output = estimation.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, None)
    measurement_times = np.array(list(filter_output.keys()))
    num_meas = len(measurement_times)
    residuals = np.zeros((num_meas,2))
    new_states = np.zeros((num_meas,6))
    new_covar = np.zeros((num_meas,6,6))
    # print(np.shape(filter_output[measurement_times[0]]['resids']))
    for i in range(num_meas):
        residuals[i,:] = filter_output[measurement_times[i]]['resids'].T
        new_states[i,:] = filter_output[measurement_times[i]]['state'].T
        new_covar[i,:,:] = filter_output[measurement_times[i]]['covar']
    residuals_full = residuals
    times_full = measurement_times

    if norad_id in boundary_dict.keys():
        boundary = boundary_dict[norad_id]
        residuals = residuals_full[boundary:]
        times = times_full[boundary:]

    RMS_res = np.sqrt(1 / num_meas * np.einsum('ij,ij->j', residuals, residuals))
    # Convert to arcsec
    RMS_res = RMS_res * 360 / 2 / np.pi * 3600

    rso.RMS_res = RMS_res
    rso.residuals_full = residuals_full
    rso.new_states = new_states
    rso.new_covar = new_covar
    rso.residuals = residuals
    rso.times_full = times_full
    rso.times = times

    rso.meas_initial_state = new_states[-1,:].reshape((6,1))
    meas_covar = new_covar[-1,:,:]
    # Remediate covariance matrix so that it is positive semi-definite
    meas_covar = util.remediate_covariance(meas_covar, 0)[0]
    rso.meas_covar = meas_covar
    rso.meas_initial_time = measurement_times[-1]
    rso.meas_trange = np.array([measurement_times[-1], tend])
    print('Final measurement time:', (measurement_times[-1] - tepoch)/3600)

    # Housekeeping data
    print('Residuals RMS:')
    print('- RA: ', RMS_res[0], 'arcsec')
    print('- Declination: ', RMS_res[1], 'arcsec')
    print('-------------UKF finished------------')
    print()



#### Add updated data from IOD analysis for RSO 99002 ####
norad_id = iod_norad_id
rso = obj_dict[norad_id]
rso.meas_initial_state = rso.cartesian_state
rso.meas_covar = rso.covar
rso.meas_initial_time = rso.epoch
rso.meas_trange = np.array([rso.epoch, tend])
obj_dict[norad_id] = rso


######## Obtain states of my_sat at the end of measurements ########
my_sat_dict = {}
my_sat_dict['nominal'] = my_sat
# Define propagation parameters
bodies_to_create = ['Sun', 'Earth', 'Moon']
rso1_params = {}
rso1_params['mass'] = my_sat.mass
rso1_params['area'] = my_sat.area
rso1_params['Cd'] = my_sat.Cd
rso1_params['Cr'] = my_sat.Cr
rso1_params['sph_deg'] = 8
rso1_params['sph_ord'] = 8
rso1_params['central_bodies'] = ['Earth']
rso1_params['bodies_to_create'] = bodies_to_create
# Define integration parameters
int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12
# Propagate my_sat to the last observation conducted for each rso or viceversa
for norad_id in obj_dict:

    if norad_id == my_norad:
        continue

    rso = obj_dict[norad_id]

    my_sat_meas_tf = my_sat.meas_initial_time
    rso_meas_tf = rso.meas_initial_time

    if my_sat_meas_tf > rso_meas_tf:
        tvec = np.array([rso_meas_tf, my_sat_meas_tf])
        rso2_params = {}
        rso2_params['mass'] = rso.mass
        rso2_params['area'] = rso.area
        rso2_params['Cd'] = rso.Cd
        rso2_params['Cr'] = rso.Cr
        rso2_params['sph_deg'] = 8
        rso2_params['sph_ord'] = 8
        rso2_params['central_bodies'] = ['Earth']
        rso2_params['bodies_to_create'] = bodies_to_create
        Xo = rso.meas_initial_state
        Po = rso.meas_covar
        # Propagate
        tf, Xf, Pf = propagate_state_and_covar(Xo, Po, tvec, rso2_params,
                                                int_params)
        rso.meas_initial_state = Xf
        # Remediate covariance matrix so that it is positive semi-definite
        Pf= util.remediate_covariance(Pf, 0)[0]
        rso.meas_covar = Pf
        rso.meas_initial_time = tf
        rso.meas_trange = np.array([tf, tend])
        my_sat_new = deepcopy(my_sat)
        my_sat_new.cartesian_state = my_sat_new.meas_initial_state
        my_sat_new.covar = util.remediate_covariance(my_sat_new.meas_covar, 0)[0]
        my_sat_dict[norad_id] = deepcopy(my_sat_new)

    elif rso_meas_tf > my_sat_meas_tf:
        # Define integration limits
        tvec = np.array([my_sat_meas_tf, rso_meas_tf])

        Xo = my_sat.meas_initial_state
        Po = my_sat.meas_covar
        # Propagate
        tf, Xf, Pf = propagate_state_and_covar(Xo, Po, tvec, rso1_params,
                                                int_params)
        my_sat_new = deepcopy(my_sat)
        my_sat_new.cartesian_state = Xf
        # Remediate covariance matrix so that it is positive semi-definite
        Pf= util.remediate_covariance(Pf, 0)[0]
        my_sat_new.covar = Pf 
        # my_sat_new.meas_tf = 0
        my_sat_dict[norad_id] = deepcopy(my_sat_new)


# Add new objects without measurements
for norad_id in new_obj_dict:
    rso = new_obj_dict[norad_id]
    my_sat_new = deepcopy(my_sat)
    my_sat_dict[norad_id] = my_sat_new
    rso.meas_initial_state = rso.cartesian_state
    rso.meas_covar = rso.covar
    rso.meas_initial_time = rso.epoch
    rso.meas_trange = np.array([rso.epoch, tend])
    if norad_id == 91762:
        rso.mass = 1.0  # Implement newfound mass from Q1
    obj_dict[norad_id] = rso

print(obj_dict.keys())

obj_dict_full = obj_dict.copy()
obj_dict.pop(my_norad)

print('Number of analysed RSOs:', len(my_sat_dict.keys()))


####################################
########## TCA Assessment ##########
####################################

print('-------------------------------')
print('TCA Assessment')
print('-------------------------------')

distance_tca = 20 * 10 ** 3  # Critical distance to identify TCAs
delete_ls = []

for norad_id in obj_dict.keys():
    my_sat = my_sat_dict[norad_id]
    obj = obj_dict[norad_id]

    # Define relevant bodies
    bodies_to_create = ['Sun', 'Earth', 'Moon']
    # Define initial cartesian states
    X1 = my_sat.cartesian_state
    X2 = obj.meas_initial_state
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
    trange = obj.meas_trange
    # print((trange-tepoch)/3600)
    T_list, rho_list = util.compute_TCA(
        X1, X2, trange, rso1_params, rso2_params,
        int_params, rho_min_crit=distance_tca)

    print('NORAD ID:', norad_id)
    print('Times for TCA [hr since epoch]:', (np.array(T_list) - tepoch) / constants.JULIAN_DAY * 24, '| rho_min [m]:',
          rho_list)
    print('Trange of analysis [hr sincew epoch]:', (trange - tepoch)/3600)
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
# 5 remaining, actually, update
print('TCA cutoff:', len(obj_dict.keys()), 'remaining')


######################################
######## Finding best TCA  ###########
######################################

bodies_to_create = ['Sun', 'Earth', 'Moon']

int_params_propagation_to_epoch = {'tudat_integrator': 'rk4', 'step': 1}

int_params_precise_analysis_forward = {'tudat_integrator': 'rkf78', 'step': 0.01, 'max_step': 0.05, 'min_step': 1e-4,
                                       'rtol': 1e-12, 'atol': 1e-12}

int_params_precise_analysis_backwards = {'tudat_integrator': 'rkf78', 'step': -0.01, 'max_step': -0.05,
                                         'min_step': -1e-4, 'rtol': 1e-12, 'atol': 1e-12}

for norad_id in obj_dict.keys():

    my_sat = my_sat_dict[norad_id]

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
    
    obj = obj_dict[norad_id]
    Xo = obj.meas_initial_state
    rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
                  'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}
    
    print('RSO :', norad_id)
    print('Propagation around estimated TCA to find better approximation')
    # First propagation to final epoch
    tvec = np.array([obj.meas_initial_time, obj.tca_T_list[0]])
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
            Xdebug = X_rso_back[i,0:3]

    for i in range(len(t_rso_for)):
        dist = np.linalg.norm(X_rso_for[i, 0:3] - X_sat_for[i, 0:3])
        if dist < min_dist:
            check = True
            min_dist = dist
            tca_real = t_rso_for[i]
            Xdebug = X_rso_for[i,0:3]

    if check is True:
        print('Final closest distance: ', min_dist, '\n')
        print('Epoch: ', tca_real, '\n')
        print('Difference from Denenberg method:\n'
              'Dist difference: ', min_dist - obj.tca_rho_list[0], '\n'
              'Time diff: ', tca_real - obj.tca_T_list[0])
        # print(Xdebug)
    else:
        print('Final closest distance and TCA coincident with Denenberg method\n'
              'Min distance: ', min_dist, '\n'
              'TCA: ', obj.tca_T_list[0])
    print('-------------------------------\n')
    obj.tca_T_list[0] = tca_real
    # print((tca_real - tepoch)/3600)
    obj.tca_rho_list[0] = min_dist



# ######################################
# ######## Propagation to TCA ##########
# ######################################

# intergator parameters
int_params = {'tudat_integrator': 'rkf78', 'step': 10., 'max_step': 1000., 'min_step': 1e-3, 'rtol': 1e-12,
              'atol': 1e-12}

for norad_id in obj_dict.keys():

    my_sat = my_sat_dict[norad_id]
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


    obj = obj_dict[norad_id]
    Xo = obj.meas_initial_state
    Po = obj.meas_covar
    Po = util.remediate_covariance(Po, 1.0E-09)[0]
    rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
                  'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}

    tf = np.zeros(len(obj.tca_T_list))
    Xf = np.zeros(shape=(6, len(obj.tca_T_list)))
    Pf = np.zeros(shape=(6, 6 * len(obj.tca_T_list)))

    prop = {'tf': np.array([]), 'Xf': np.array([]), 'Pf': np.array([])}

    for i in range(len(obj.tca_T_list)):
        tvec = np.array([obj.meas_initial_time, obj.tca_T_list[i]])
        # print(norad_id, rso_params)
        tf[i], Xf_new, Pf_matrix = propagate_state_and_covar(Xo, Po, tvec, rso_params, int_params)
        # t_rso, X_rso_prop = propagate_orbit(Xo, tvec, rso_params, int_params)
        # if norad_id == 91762:
        #     print(t_rso[-1], tf[i])
        #     print(X_rso_prop[-1,:3], Xf_new[:3])
        Pf_matrix = util.remediate_covariance(Pf_matrix, 0)[0]
        Xf[:, i] = Xf_new.reshape((6,))
        inx = 6 * i
        Pf[:, inx:(inx + 6)] = Pf_matrix

        tvec_sat = np.array([obj.meas_initial_time, obj.tca_T_list[i]])
        tf_sat, Xf_sat, Pf_matrix_sat = propagate_state_and_covar(Xo_sat, Po_sat, tvec_sat, sat_params, int_params)
        Pf_matrix_sat = util.remediate_covariance(Pf_matrix_sat, 0)[0]

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

    my_sat = my_sat_dict[norad_id]
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


    file_path = write_cdm(obj.utc, obj.tca_T_list[0], d, M, np.linalg.norm(rel_vel), rel_pos_ric,
                         Pc, norad_id, Xf_sat, Xf_rso, Pf_sat, Pf_rso)

    # print('CDM generated and saved in: ', file_path)
    print('\n-------------------------------')
    print('-------------------------------')

########################################
######## Data for NASA MonteCarlo ######
########################################

montecarlo_data_dir = os.path.join(current_dir, 'CARA_matlab', 'MonteCarloPc', 'data_files')
obj_filename = os.path.join(montecarlo_data_dir, 'obj_dict.txt')

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

    my_sat = my_sat_dict[norad_id]
    Xf_sat = getattr(my_sat, norad_id_str)['Xf']
    Pf_sat = getattr(my_sat, norad_id_str)['Pf']
    matlab_data[i, 44] = my_sat.NORAD_ID
    matlab_data[i, 45] = my_sat.area
    matlab_data[i, 46:52] = Xf_sat[:, 0]
    matlab_data[i, 52:88] = Pf_sat.flatten()

    i += 1

np.savetxt(obj_filename, matlab_data)


########################################
######## Plot creation #################
########################################

###################################################################################
# Plot the point distribution at the end of the measurements
bodies_to_create = ['Sun', 'Earth', 'Moon']
obj = obj_dict[91883]
Xo = obj.cartesian_state
Po = obj.covar
int_params = {'tudat_integrator': 'rkf78', 'step': 10., 'max_step': 1000., 'min_step': 1e-3, 'rtol': 1e-12,
              'atol': 1e-12}
tvec = np.array([obj.epoch, obj.meas_initial_time])
rso2_params = {}
rso2_params['mass'] = obj.mass
rso2_params['area'] = obj.area
rso2_params['Cd'] = obj.Cd
rso2_params['Cr'] = obj.Cr
rso2_params['sph_deg'] = 8
rso2_params['sph_ord'] = 8
rso2_params['central_bodies'] = ['Earth']
rso2_params['bodies_to_create'] = bodies_to_create
tf, Xf, Pf = propagate_state_and_covar(Xo, Po, tvec, rso2_params,
                                                int_params)
Pf = util.remediate_covariance(Pf,1.0E-10)[0]
obj.propagated_state = Xf 
obj.propagated_covar = Pf
norad_id = obj.NORAD_ID
fig = plot_gen.position_distribution(obj)
fpath = os.path.join(current_dir, 'output','plots','q4', 'position_distribution', str(norad_id) + '_position_distribution.png')
fig.savefig(fpath)
fig = plot_gen.position_distribution_propagated(obj)
fpath = os.path.join(current_dir, 'output','plots','q4', 'position_distribution', str(norad_id) + '_position_distribution_propagated.png')
fig.savefig(fpath)
fig = plot_gen.position_distribution_measured(obj)
fpath = os.path.join(current_dir, 'output','plots','q4', 'position_distribution', str(norad_id) + '_position_distribution_measured.png')
fig.savefig(fpath)
#######################################################################################################

#### Plot the residuals in time ####
residuals_ls = ['RA', 'DEC']
for norad_id in close_encounter_ls:
    rso = obj_dict_full[norad_id]
    for residual_type in residuals_ls:
        fig = plot_gen.residuals_full(rso, boundary_dict, residual_type)
        fpath = os.path.join(current_dir, 'output','plots','q4', 'residuals',  str(norad_id)+'_' + residual_type + '_residuals_full.png')
        fig.savefig(fpath)

        fig = plot_gen.residuals_cut(rso, boundary_dict, residual_type)
        fpath = os.path.join(current_dir, 'output','plots','q4', 'residuals', str(norad_id)+'_' + residual_type + '_residuals_cut.png')
        fig.savefig(fpath)

        plt.close()

#### Plot the RIC covariance in time ####
for norad_id in close_encounter_ls:

    rso = obj_dict_full[norad_id]
    fig = plot_gen.covar_ric(rso)
    fpath = os.path.join(current_dir, 'output','plots','q4', 'ric', str(norad_id) + '_ric_covar.png')
    fig.savefig(fpath)






