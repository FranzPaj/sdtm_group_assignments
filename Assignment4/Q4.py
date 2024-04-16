#### Code for Question 4 ####

import numpy as np
import os
import inspect
from math import ceil, log
from pycode.CDM import write_cdm
import pickle
import matplotlib.pyplot as plt

from tudatpy import constants
import pycode.ConjunctionUtilities as util
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
num_objs = len(data_dict.keys())



# Define a dictionary of relevant possible impactors as class objects
# close_encounter_ls = [99002, 91332, 91395, 91509, 91686, 91883]
close_encounter_ls = [91332]
#### Is 99002 a new one? What is that? Check

obj_dict = dict()  # Object dictionary
for elem in data_dict.keys():
    if elem in close_encounter_ls:
        # Each element of the possible impactors dictionary is an object with useful properties
        Obj = util.Object(data_dict, elem)
        obj_dict[elem] = Obj
# Add our object
my_norad = 40940
my_sat = util.Object(data_dict, my_norad)
# obj_dict[my_norad] = my_sat  # ADD LATER, COMMENTED ONLY FOR DEBUGGING

# Define time interval of interest for collision analysis
tepoch = my_sat.epoch
tspan = 2 * constants.JULIAN_DAY
trange = np.array([tepoch, tepoch + tspan])

################################################
########## Elaborate new measurements ##########
################################################

# Define parameters for filter and propagation
int_params = {'tudat_integrator': 'rkf78',
              'step': 10.,
              'max_step': 1000.,
              'min_step': 1e-3,
              'rtol': 1e-12,
              'atol': 1e-12}

# Extract estimation from measurements using a UKF
for rso in obj_dict:

    state_params, meas_dict, sensor_params = estimation.read_measurement_file(
        os.path.join(current_dir, 'data', 'states_updated', 'q4_meas_rso_'+str(rso)+'.pkl'))
    
    # # Determine biggest gap
    # t_opt = np.array(meas_dict['tk_list'])
    # gap = np.max(np.diff(t_opt))

    # Define RSO-specific parameters for the filter
    # Optical 1
    # filter_params = {
    #     'Qeci': 0 * np.diag([1., 1., 1.]),
    #     'Qric': 8.0E-15 * np.diag([20., 1., 1.]),
    #     'alpha': 1e-4,
    #     'gap_seconds': 1.0E10
    # }
    # Optical 2
    filter_params = {
        'Qeci': 0 * np.diag([1., 1., 1.]),
        'Qric': 5.0E-15 * np.diag([20., 1., 1.]),
        'alpha': 1.0E-3,
        'gap_seconds': 1.0E10
    }

    filter_output = estimation.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, None)
    measurement_times = np.array(list(filter_output.keys()))
    num_meas = len(measurement_times)
    residuals = np.zeros((num_meas,2))
    # print(np.shape(filter_output[measurement_times[0]]['resids']))
    for i in range(num_meas):
        residuals[i,:] = filter_output[measurement_times[i]]['resids'].T
    RMS_res = np.sqrt(1 / num_meas * np.einsum('ij,ij->j', residuals, residuals))

    plt.scatter(measurement_times,residuals[:,0])
    plt.show()
    plt.scatter(measurement_times,residuals[:,1])
    plt.show()

    # Get new estimate for initial state
    # estimated_state = filter_output['state']
    # estimated_covar = filter_output['covar']

    

####### Also ADD THE NEW OBJECTS!!!









# ####################################
# ########## TCA Assessment ##########
# ####################################

# print('-------------------------------')
# print('TCA Assessment')
# print('-------------------------------')

# distance_tca = 20 * 10 ** 3  # Critical distance to identify TCAs
# delete_ls = []

# for norad_id in obj_dict.keys():

#     obj = obj_dict[norad_id]
#     # Define relevant bodies
#     bodies_to_create = ['Sun', 'Earth', 'Moon']
#     # bodies = prop.tudat_initialize_bodies(bodies_to_create)

#     # Define initial cartesian states
#     X1 = my_sat.cartesian_state
#     X2 = obj.cartesian_state
#     # Define state params
#     # rso1 state params
#     rso1_params = {}
#     rso1_params['mass'] = my_sat.mass
#     rso1_params['area'] = my_sat.area
#     rso1_params['Cd'] = my_sat.Cd
#     rso1_params['Cr'] = my_sat.Cr
#     rso1_params['sph_deg'] = 8
#     rso1_params['sph_ord'] = 8
#     rso1_params['central_bodies'] = ['Earth']
#     rso1_params['bodies_to_create'] = bodies_to_create
#     # rso2 state params
#     rso2_params = {}
#     rso2_params['mass'] = obj.mass
#     rso2_params['area'] = obj.area
#     rso2_params['Cd'] = obj.Cd
#     rso2_params['Cr'] = obj.Cr
#     rso2_params['sph_deg'] = 8
#     rso2_params['sph_ord'] = 8
#     rso2_params['central_bodies'] = ['Earth']
#     rso2_params['bodies_to_create'] = bodies_to_create
#     # Define integration parameters
#     int_params = {}
#     int_params['tudat_integrator'] = 'rkf78'
#     int_params['step'] = 10.
#     int_params['max_step'] = 1000.
#     int_params['min_step'] = 1e-3
#     int_params['rtol'] = 1e-12
#     int_params['atol'] = 1e-12
#     # Find TCA
#     T_list, rho_list = util.compute_TCA(
#         X1, X2, trange, rso1_params, rso2_params,
#         int_params, rho_min_crit=distance_tca)

#     print('NORAD ID:', norad_id)
#     print('Times for TCA [hr since epoch]:', (np.array(T_list) - tepoch) / constants.JULIAN_DAY * 24, '| rho_min [m]:',
#           rho_list)
#     print('-------------------------------')

#     # Identify possible HIEs
#     if min(rho_list) < distance_tca:
#         obj.tca_T_list = T_list
#         obj.tca_rho_list = rho_list
#     else:
#         delete_ls.append(norad_id)

# # Eliminate impossible impactors
# for norad_id in delete_ls:
#     obj_dict.pop(norad_id)
# # 5 remaining
# print('TCA cutoff:', len(obj_dict.keys()), 'remaining')

# ######################################
# ######## Finding best TCA  ###########
# ######################################
# bodies_to_create = ['Sun', 'Earth', 'Moon']

# sat_params = {}
# sat_params['mass'] = my_sat.mass
# sat_params['area'] = my_sat.area
# sat_params['Cd'] = my_sat.Cd
# sat_params['Cr'] = my_sat.Cr
# sat_params['sph_deg'] = 8
# sat_params['sph_ord'] = 8
# sat_params['central_bodies'] = ['Earth']
# sat_params['bodies_to_create'] = bodies_to_create
# Xo_sat = my_sat.cartesian_state
# Po_sat = my_sat.covar

# int_params_propagation_to_epoch = {'tudat_integrator': 'rk4', 'step': 1}

# int_params_precise_analysis_forward = {'tudat_integrator': 'rkf78', 'step': 0.01, 'max_step': 0.05, 'min_step': 1e-4,
#                                        'rtol': 1e-12, 'atol': 1e-12}

# int_params_precise_analysis_backwards = {'tudat_integrator': 'rkf78', 'step': -0.01, 'max_step': -0.05,
#                                          'min_step': -1e-4, 'rtol': 1e-12, 'atol': 1e-12}

# for norad_id in obj_dict.keys():
#     obj = obj_dict[norad_id]
#     Xo = obj.cartesian_state
#     rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
#                   'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}

#     print('RSO :', norad_id)
#     print('Propagation around estimated TCA to find better approximation')
#     # First propagation to final epoch
#     tvec = np.array([obj.epoch, obj.tca_T_list[0]])
#     t_rso, X_rso_prop = propagate_orbit(Xo, tvec, rso_params, int_params_propagation_to_epoch)
#     print('RSO normal prop done')
#     t_sat, X_sat_prop = propagate_orbit(Xo_sat, tvec, sat_params, int_params_propagation_to_epoch)
#     print('SAT normal prop done')

#     # Propagation around epoch
#     end_time = obj.tca_T_list[0] - 10
#     tvec_backwards = np.array([obj.tca_T_list[0], end_time])
#     t_rso_back, X_rso_back = propagate_orbit(X_rso_prop[-1, :], tvec_backwards, rso_params,
#                                              int_params_precise_analysis_backwards)
#     print('RSO backwards prop done')
#     t_sat_back, X_sat_back = propagate_orbit(X_sat_prop[-1, :], tvec_backwards, sat_params,
#                                              int_params_precise_analysis_backwards)
#     print('SAT backwards prop done')
#     tvec_forward = np.array([obj.tca_T_list[0], obj.tca_T_list[0] + 10])
#     t_rso_for, X_rso_for = propagate_orbit(X_rso_prop[-1, :], tvec_forward, rso_params,
#                                            int_params_precise_analysis_forward)
#     print('RSO forward prop done')
#     t_sat_for, X_sat_for = propagate_orbit(X_sat_prop[-1, :], tvec_forward, sat_params,
#                                            int_params_precise_analysis_forward)
#     print('SAT backwards prop done')

#     check = False
#     tca_real = obj.tca_T_list[0]
#     min_dist = obj.tca_rho_list[0]
#     for i in range(len(t_rso_back)):
#         dist = np.linalg.norm(X_rso_back[i, 0:3] - X_sat_back[i, 0:3])
#         if dist < min_dist:
#             check = True
#             min_dist = dist
#             tca_real = t_rso_back[i]

#     for i in range(len(t_rso_for)):
#         dist = np.linalg.norm(X_rso_for[i, 0:3] - X_sat_for[i, 0:3])
#         if dist < min_dist:
#             check = True
#             min_dist = dist
#             tca_real = t_rso_for[i]

#     if check is True:
#         print('Final closest distance: ', min_dist, '\n')
#         print('Epoch: ', tca_real, '\n')
#         print('Difference from Denenberg method:\n'
#               'Dist difference: ', min_dist - obj.tca_rho_list[0], '\n'
#               'Time diff: ', tca_real - obj.tca_T_list[0])
#     else:
#         print('Final closest distance and TCA coincident with Denenberg method\n'
#               'Min distance: ', min_dist, '\n'
#               'TCA: ', obj.tca_T_list[0])
#     print('-------------------------------\n')
#     obj.tca_T_list[0] = tca_real
#     obj.tca_rho_list[0] = min_dist



# ######################################
# ######## Propagation to TCA ##########
# ######################################
# # intergator parameters
# int_params = {'tudat_integrator': 'rkf78', 'step': 10., 'max_step': 1000., 'min_step': 1e-3, 'rtol': 1e-12,
#               'atol': 1e-12}

# for norad_id in obj_dict.keys():

#     obj = obj_dict[norad_id]
#     Xo = obj.cartesian_state
#     Po = obj.covar

#     rso_params = {'mass': obj.mass, 'area': obj.area, 'Cd': obj.Cd, 'Cr': obj.Cr, 'sph_deg': 8,
#                   'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': bodies_to_create}

#     tf = np.zeros(len(obj.tca_T_list))
#     Xf = np.zeros(shape=(6, len(obj.tca_T_list)))
#     Pf = np.zeros(shape=(6, 6 * len(obj.tca_T_list)))

#     prop = {'tf': np.array([]), 'Xf': np.array([]), 'Pf': np.array([])}

#     for i in range(len(obj.tca_T_list)):
#         tvec = np.array([obj.epoch, obj.tca_T_list[i]])
#         tf[i], Xf_new, Pf_matrix = propagate_state_and_covar(Xo, Po, tvec, rso_params, int_params)
#         Xf[:, i] = Xf_new.reshape((6,))
#         inx = 6 * i
#         Pf[:, inx:(inx + 6)] = Pf_matrix

#         tvec_sat = np.array([my_sat.epoch, obj.tca_T_list[i]])
#         tf_sat, Xf_sat, Pf_matrix_sat = propagate_state_and_covar(Xo_sat, Po_sat, tvec_sat, sat_params, int_params)

#         if len(prop['tf']) == 0:
#             prop['tf'] = tf_sat
#         else:
#             prop['tf'] = [prop['tf'], tf_sat]

#         if len(prop['Xf']) == 0:
#             prop['Xf'] = Xf_sat
#         else:
#             prop['Xf'] = np.column_stack((prop['Xf'], Xf_sat))

#         if len(prop['Pf']) == 0:
#             prop['Pf'] = Pf_matrix_sat
#         else:
#             prop['Pf'] = np.column_stack((prop['Pf'], Pf_matrix_sat))

#     # print(my_sat.norad_id)
#     obj.tf = tf
#     obj.Xf = Xf
#     obj.Pf = Pf

#     setattr(my_sat, str(norad_id), prop)

# #######################################
# ######## Detailed assessment ##########
# #######################################
# print('-------------------------------')
# print('Detailed assessment')
# print('-------------------------------')

# for norad_id in obj_dict.keys():
#     obj = obj_dict[norad_id]
#     norad_id_str = str(norad_id)
#     print('RSO :', norad_id_str)

#     Xf_sat = getattr(my_sat, norad_id_str)['Xf']
#     Pf_sat = getattr(my_sat, norad_id_str)['Pf']
#     # print(Pf_sat[0:3, 0:3])

#     Xf_rso = obj.Xf
#     Pf_rso = obj.Pf

#     HBR = np.sqrt(my_sat.area / np.pi) + np.sqrt(obj.area / np.pi)
#     Pc = util.Pc2D_Foster(Xf_sat, Pf_sat, Xf_rso, Pf_rso, HBR, rtol=1e-8, HBR_type='circle')

#     M = util.compute_mahalanobis_distance(Xf_sat[0:3], Xf_rso[0:3], Pf_sat[0:3, 0:3], Pf_rso[0:3, 0:3])
#     d = util.compute_euclidean_distance(Xf_sat[0:3], Xf_rso[0:3])

#     print('Mahalanobis distance:\t\t\t\t ', M)
#     print('Foster Probability of collision:\t ', Pc)
#     # print('Euclidean distance:\t ', d)

#     ''' Broken for now (Pc not psd)
#     if Pc > 1e-6:
#         N_Mc = 10**(ceil(log(1/Pc * 10, 10)))
#         print('Number of points for Montecarlo analysis:\t', N_Mc)

#         P_mc = util.montecarlo_Pc_final(Xf_sat[0:3], Xf_rso[0:3], Pf_sat[0:3, 0:3], Pf_rso[0:3, 0:3], HBR, N=N_Mc)
#         print('Montecarlo Probability of collision:\t', P_mc)
#     '''

#     Xf_sat_ric = util.eci2ric(Xf_sat[0:3], Xf_sat[3:6], Xf_sat[0:3])
#     Xf_obj_ric = util.eci2ric(Xf_sat[0:3], Xf_sat[3:6], Xf_rso[0:3])
#     print('Radial distance:\t\t\t\t ', np.abs(Xf_sat_ric[0] - Xf_obj_ric[0])[0])
#     print('Along Track distance:\t\t\t ', np.abs(Xf_obj_ric[1])[0])
#     print('Cross Track distance:\t\t\t ', np.abs(Xf_obj_ric[2])[0])

#     print('USEFUL INFORMATION')
#     rel_pos = Xf_rso[0:3] - Xf_sat[0:3]
#     rel_vel = Xf_rso[3:6] - Xf_sat[3:6]
#     rel_pos_ric = util.eci2ric(Xf_sat[0:3], Xf_sat[3:6], rel_pos)
#     rel_vel_ric = util.eci2ric_vel(Xf_sat[0:3], Xf_sat[3:6], rel_pos_ric, rel_vel)
#     print('Relative Velocity in ECI:\n', rel_vel)
#     print('Relative Velocity in RSW:\n', rel_vel_ric)


#     filename = write_cdm(obj.utc, obj.tca_T_list[0], d, M, np.linalg.norm(rel_vel), rel_pos_ric,
#                          Pc, norad_id, Xf_sat, Xf_rso, Pf_sat, Pf_rso)

#     print('CDM generated and saved in: ', filename)
#     print('\n-------------------------------')
#     print('-------------------------------')

# ########################################
# ######## Data for NASA MonteCarlo ######
# ########################################

# montecarlo_data_dir = os.path.join(current_dir, 'CARA_matlab', 'MonteCarloPc', 'data_files')
# obj_filename = os.path.join(montecarlo_data_dir, 'obj_dict_test.txt')

# matlab_data = np.zeros((len(obj_dict.keys()), 88))
# i = 0
# # Change into comprehensible format for matlab:
# for norad_id in obj_dict.keys():
#     obj = obj_dict[norad_id]
#     matlab_data[i, 0] = norad_id
#     matlab_data[i, 1] = obj.area
#     matlab_data[i, 2:8] = obj.Xf[:, 0]
#     matlab_data[i, 8:44] = obj.Pf.flatten()

#     norad_id_str = str(norad_id)
#     Xf_sat = getattr(my_sat, norad_id_str)['Xf']
#     Pf_sat = getattr(my_sat, norad_id_str)['Pf']
#     matlab_data[i, 44] = my_sat.NORAD_ID
#     matlab_data[i, 45] = my_sat.area
#     matlab_data[i, 46:52] = Xf_sat[:, 0]
#     matlab_data[i, 52:88] = Pf_sat.flatten()

#     i += 1

# np.savetxt(obj_filename, matlab_data)

















# import pycode.EstimationUtilities as util
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# # Extract measurements
# state_opt_a, meas_opt_a, sensor_params_opt_a = util.read_measurement_file(
#     './data/group2/group2_q2a_skymuster_sparse_optical_meas.pkl')

# state_rad_a, meas_rad_a, sensor_params_rad_a = util.read_measurement_file(
#     './data/group2/group2_q2a_skymuster_sparse_radar_meas.pkl')

# state_opt_b, meas_opt_b, sensor_params_opt_b = util.read_measurement_file(
#     './data/group2/group2_q2b_skymuster_sparse_optical_meas.pkl')

# state_rad_b, meas_rad_b, sensor_params_rad_b = util.read_measurement_file(
#     './data/group2/group2_q2b_skymuster_sparse_radar_meas.pkl')

# t_truth_a, state_truth_a, params_truth_a = util.read_truth_file(
#     './data/group2/group2_q2a_skymuster_sparse_truth_grav.pkl')

# t_truth_b, state_truth_b, params_truth_b = util.read_truth_file(
#     './data/group2/group2_q2b_skymuster_sparse_truth_grav_srp.pkl')

# bodies = state_opt_a['bodies_to_create']
# ################################# POINT A #####################################
# # Develop a procedure to adjust the SNC covariance to process the measurements
# # using the UKF. State your final selection with justification. You must
# # include, at minimum, a plot of your position errors with 3-σ covariance
# # bounds in the Radial-Intrack-Crosstrack (RIC) coordinate frame, as well as
# # a plot of post-fit measurement residuals using your selected SNC parameters.
# # You may include any additional plots or tables at your discretion.


# # ukf(state_params, meas_dict, sensor_params,
# # int_params, filter_params, bodies)
# # where filter_params: Q_eci and Q_ric


# # Tune biggest gap in seconds between measurements
# t_opt_a = np.asarray(meas_opt_a['tk_list'])
# t_opt_b = np.asarray(meas_opt_b['tk_list'])
# t_rad_a = np.asarray(meas_rad_a['tk_list'])
# t_rad_b = np.asarray(meas_rad_b['tk_list'])
# time_gaps = np.diff(t_opt_a)
# gap_truth_a = np.max(np.diff(t_truth_a))
# gap_truth_b = np.max(np.diff(t_truth_b))
# gap_opt_a = np.max(np.diff(t_opt_a))
# gap_opt_b = np.max(np.diff(t_opt_b))
# gap_rad_a = np.max(np.diff(t_rad_a))
# gap_rad_b = np.max(np.diff(t_rad_b))

# gap_seconds = 6*3600                  # to tune

# int_params = {'tudat_integrator': 'rkf78',
#               'step': 10.,
#               'max_step': 1000.,
#               'min_step': 1e-3,
#               'rtol': 1e-12,
#               'atol': 1e-12}


# def analysis(truth_state, state, meas,
#              sensor, integ, filters, time, name: str, mtype, plotting):
#     rad2arcsec = 1 / ((1. / 3600.) * np.pi / 180.)
#     filter_output = util.ukf(state, meas, sensor,
#                              integ, filters, None)
#     # extract covars and state
#     covars = np.vstack(
#         [(filter_output[t])['covar'].reshape(1, 36) for t in time])
#     state = np.vstack(
#         [(filter_output[t])['state'].reshape(1, 6) for t in time])
#     dummy = len((filter_output[time[0]])['resids'])
#     resid = np.vstack(
#         [(filter_output[t])['resids'].reshape(1, dummy) for t in time])

#     error_ric = np.zeros((len(time), 3))

#     error = truth_state - state

#     cov_pos_ric = np.zeros((len(time), 9))

#     for i in range(len(time)):
#         # int state
#         error_ric[i, :] = util.eci2ric(
#             state[i, 0:3], state[i, 3:6], error[i, :3]).reshape(1, 3)
#         cov = covars[i, :].reshape(6, 6)
#         cov_pos_ric[i, :] = (util.eci2ric(
#             state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)

#     # sigma bound
#     sigma_bound_pos = 3 * np.sqrt(cov_pos_ric)

#     RMS_pos = np.sqrt(np.sum(error[:, 0:3]**2)/len(time))
#     RMS_vel = np.sqrt(np.sum(error[:, 3:6]**2)/len(time))
#     if mtype == 'opt':
#         RMS_RA = np.sqrt(np.sum((resid[:, 0]*rad2arcsec) ** 2) / len(time))
#         RMS_DEC = np.sqrt(np.sum((resid[:, 1]*rad2arcsec) ** 2) / len(time))
#         print(RMS_RA)
#         print(RMS_DEC)
#     if mtype == 'rad':
#         RMS_range = np.sqrt(np.sum(resid[:, 0] ** 2) / len(time))
#         RMS_RA = np.sqrt(np.sum((resid[:, 1]*rad2arcsec) ** 2) / len(time))
#         RMS_DEC = np.sqrt(np.sum((resid[:, 2]*rad2arcsec) ** 2) / len(time))
#         print(RMS_range)
#         print(RMS_RA)
#         print(RMS_DEC)

#     if plotting:
#         plt.rc('axes', titlesize=16)  # fontsize of the axes title
#         plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
#         plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
#         plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
#         plt.rc('legend', fontsize=14)  # legend fontsize
#         # plotting 1
#         time_rel = (time - time[0]) / 3600
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#         ax1.plot(time_rel, error_ric[:, 0])
#         ax1.plot(time_rel, sigma_bound_pos[:, 0], 'k:')
#         ax1.plot(time_rel, -sigma_bound_pos[:, 0], 'k:')
#         ax1.grid()
#         ax2.plot(time_rel, error_ric[:, 1])
#         ax2.plot(time_rel, sigma_bound_pos[:, 4], 'k:')
#         ax2.plot(time_rel, -sigma_bound_pos[:, 4], 'k:')
#         ax2.grid()
#         ax3.plot(time_rel, error_ric[:, 2])
#         ax3.plot(time_rel, sigma_bound_pos[:, 8], 'k:')
#         ax3.plot(time_rel, -sigma_bound_pos[:, 8], 'k:')
#         ax3.grid()
#         plt.tight_layout()
#         plt.savefig('plots/' + name + '_error.png')

#         fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
#               (ax9, ax10, ax11, ax12)) = plt.subplots(
#             3, 4, figsize=(12, 8))
#         # R direction
#         ax1.plot(time_rel[:31], error_ric[:31, 0])
#         ax1.plot(time_rel[:31], sigma_bound_pos[:31, 0], 'k:')
#         ax1.plot(time_rel[:31], -sigma_bound_pos[:31, 0], 'k:')
#         ax1.grid()
#         ax1.set_ylabel('R-direction [m]', size=18)
#         ax2.plot(time_rel[31:62], error_ric[31:62, 0])
#         ax2.plot(time_rel[31:62], sigma_bound_pos[31:62, 0], 'k:')
#         ax2.plot(time_rel[31:62], -sigma_bound_pos[31:62, 0], 'k:')
#         ax2.grid()
#         ax3.plot(time_rel[62:93], error_ric[62:93, 0])
#         ax3.plot(time_rel[62:93], sigma_bound_pos[62:93, 0], 'k:')
#         ax3.plot(time_rel[62:93], -sigma_bound_pos[62:93, 0], 'k:')
#         ax3.grid()
#         ax4.plot(time_rel[93:], error_ric[93:, 0])
#         ax4.plot(time_rel[93:], sigma_bound_pos[93:, 0], 'k:')
#         ax4.plot(time_rel[93:], -sigma_bound_pos[93:, 0], 'k:')
#         ax4.grid()
#         # I direction
#         ax5.plot(time_rel[:31], error_ric[:31, 1])
#         ax5.set_ylabel('I-direction [m]', size=18)
#         ax5.plot(time_rel[:31], sigma_bound_pos[:31, 4], 'k:')
#         ax5.plot(time_rel[:31], -sigma_bound_pos[:31, 4], 'k:')
#         ax5.grid()
#         ax6.plot(time_rel[31:62], error_ric[31:62, 1])
#         ax6.plot(time_rel[31:62], sigma_bound_pos[31:62, 4], 'k:')
#         ax6.plot(time_rel[31:62], -sigma_bound_pos[31:62, 4], 'k:')
#         ax6.grid()
#         ax7.plot(time_rel[62:93], error_ric[62:93, 1])
#         ax7.plot(time_rel[62:93], sigma_bound_pos[62:93, 4], 'k:')
#         ax7.plot(time_rel[62:93], -sigma_bound_pos[62:93, 4], 'k:')
#         ax7.grid()
#         ax8.plot(time_rel[93:], error_ric[93:, 1])
#         ax8.plot(time_rel[93:], sigma_bound_pos[93:, 4], 'k:')
#         ax8.plot(time_rel[93:], -sigma_bound_pos[93:, 4], 'k:')
#         ax8.grid()
#         # C direction
#         ax9.plot(time_rel[:31], error_ric[:31, 2])
#         ax9.plot(time_rel[:31], sigma_bound_pos[:31, 8], 'k:')
#         ax9.plot(time_rel[:31], -sigma_bound_pos[:31, 8], 'k:')
#         ax9.set_ylabel('C-direction [m]', size=18)
#         ax9.set_xlabel('Time [hours]', size=18)
#         ax9.grid()
#         ax10.plot(time_rel[31:62], error_ric[31:62, 2])
#         ax10.plot(time_rel[31:62], sigma_bound_pos[31:62, 8], 'k:')
#         ax10.plot(time_rel[31:62], -sigma_bound_pos[31:62, 8], 'k:')
#         ax10.set_xlabel('Time [hours]', size=18)
#         ax10.grid()
#         ax11.plot(time_rel[62:93], error_ric[62:93, 2])
#         ax11.plot(time_rel[62:93], sigma_bound_pos[62:93, 8], 'k:')
#         ax11.plot(time_rel[62:93], -sigma_bound_pos[62:93, 8], 'k:')
#         ax11.set_xlabel('Time [hours]', size=18)
#         ax11.grid()
#         ax12.plot(time_rel[93:], error_ric[93:, 2])
#         ax12.plot(time_rel[93:], sigma_bound_pos[93:, 8], 'k:')
#         ax12.plot(time_rel[93:], -sigma_bound_pos[93:, 8], 'k:')
#         ax12.set_xlabel('Time [hours]', size=18)
#         ax12.grid()
#         plt.tight_layout()
#         plt.savefig('plots/' + name + '_error_sparse.png')

#         if mtype == 'rad':

#             fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
#                 (ax9, ax10, ax11, ax12)) = plt.subplots(
#                 3, 4, figsize=(12, 8))
#             # R direction
#             ax1.scatter(time_rel[:31], resid[:31, 0])
#             ax1.grid()
#             ax1.set_ylabel('Range [m]', size=18)
#             ax2.scatter(time_rel[31:62], resid[31:62, 0])
#             ax2.grid()
#             ax3.scatter(time_rel[62:93], resid[62:93, 0])
#             ax3.grid()
#             ax4.scatter(time_rel[93:], resid[93:, 0])
#             ax4.grid()
#             # I direction
#             ax5.scatter(time_rel[:31], resid[:31, 1]*rad2arcsec)
#             ax5.set_ylabel('RA [arcsec]', size=18)
#             ax5.grid()
#             ax6.scatter(time_rel[31:62], resid[31:62, 1]*rad2arcsec)
#             ax6.grid()
#             ax7.scatter(time_rel[62:93], resid[62:93, 1]*rad2arcsec)
#             ax7.grid()
#             ax8.scatter(time_rel[93:], resid[93:, 1]*rad2arcsec)
#             ax8.grid()
#             # C direction
#             ax9.scatter(time_rel[:31], resid[:31, 2]*rad2arcsec)
#             ax9.set_ylabel('DEC [arcsec]', size=18)
#             ax9.set_xlabel('Time [hours]', size=18)
#             ax9.grid()
#             ax10.scatter(time_rel[31:62], resid[31:62, 2]*rad2arcsec)
#             ax10.set_xlabel('Time [hours]', size=18)
#             ax10.grid()
#             ax11.scatter(time_rel[62:93], resid[62:93, 2]*rad2arcsec)
#             ax11.set_xlabel('Time [hours]', size=18)
#             ax11.grid()
#             ax12.scatter(time_rel[93:], resid[93:, 2]*rad2arcsec)
#             ax12.set_xlabel('Time [hours]', size=18)
#             ax12.grid()
#             plt.tight_layout()
#             plt.savefig('plots/' + name + '_residuals.png')

#         elif mtype == 'opt':
#             rad2arcsec = 3600. * 180. / np.pi
#             fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
#                 2, 4, figsize=(16, 10))
#             # R direction
#             ax1.scatter(time_rel[:31], resid[:31, 0] * rad2arcsec)
#             ax1.set_ylabel('RA [arcsec]', size=18)
#             ax1.grid()
#             ax2.scatter(time_rel[31:62], resid[31:62, 0] * rad2arcsec)
#             ax2.grid()
#             ax3.scatter(time_rel[62:93], resid[62:93, 0] * rad2arcsec)
#             ax3.grid()
#             ax4.scatter(time_rel[93:], resid[93:, 0] * rad2arcsec)
#             ax4.grid()
#             # I direction
#             ax5.scatter(time_rel[:31], resid[:31, 1] * rad2arcsec)
#             ax5.set_ylabel('DEC [arcsec]', size=18)
#             ax5.set_xlabel('Time [hours]', size=18)
#             ax5.grid()
#             ax6.scatter(time_rel[31:62], resid[31:62, 1] * rad2arcsec)
#             ax6.set_xlabel('Time [hours]', size=18)
#             ax6.grid()
#             ax7.scatter(time_rel[62:93], resid[62:93, 1] * rad2arcsec)
#             ax7.set_xlabel('Time [hours]', size=18)
#             ax7.grid()
#             ax8.scatter(time_rel[93:], resid[93:, 1] * rad2arcsec)
#             ax8.set_xlabel('Time [hours]', size=18)
#             ax8.grid()
#             plt.tight_layout()
#             plt.savefig('plots/' + name + '_residuals.png')

#     return RMS_pos, RMS_vel


# # Radar A
# # Set process noise
# # Qeci = 0 * np.diag([1., 1., 1.])
# # Qric = 8e-15 * np.diag([1., 1., 1.])
# # alpha = 1e-4                            # to tune (in range [1e-4, 1])
# # filter_params = {
# #     'Qeci': Qeci,
# #     'Qric': Qric,
# #     'alpha': alpha,
# #     'gap_seconds': gap_seconds
# # }
# # rms_rad_pos_a, rms_rad_vel_a = analysis(
# #     state_truth_a, state_rad_a, meas_rad_a, sensor_params_rad_a,
# #     int_params, filter_params, t_rad_a, 'radar_a', 'rad', 1)
# # print(rms_rad_pos_a)
# # print(rms_rad_vel_a)
# # # Optical A
# # Set process noise
# # Qeci = 1e-9 * np.diag([1., 1., 1.])
# # Qric = 0 * np.diag([20., 1., 1.])
# # alpha = 1e-4                            # to tune (in range [1e-4, 1])
# # filter_params = {
# #     'Qeci': Qeci,
# #     'Qric': Qric,
# #     'alpha': alpha,
# #     'gap_seconds': gap_seconds
# # }
# # rms_opt_pos_a, rms_opt_vel_a = analysis(
# #     state_truth_a, state_opt_a, meas_opt_a, sensor_params_opt_a,
# #     int_params, filter_params, t_opt_a, 'optical_a', 'opt', 1)
# # print(rms_opt_pos_a)
# # print(rms_opt_vel_a)

# # Radar B
# # Set process noise
# Qeci = 0 * np.diag([1., 1., 1.])
# Qric = 8e-12 * np.diag([1., 1., 1.])
# alpha = 1e-4                            # to tune (in range [1e-4, 1])
# filter_params = {
#     'Qeci': Qeci,
#     'Qric': Qric,
#     'alpha': alpha,
#     'gap_seconds': gap_seconds
# }
# rms_rad_pos_b, rms_rad_vel_b = analysis(
#     state_truth_b, state_rad_b, meas_rad_b, sensor_params_rad_b,
#     int_params, filter_params, t_rad_b, 'radar_b', 'rad', 1)
# print(rms_rad_pos_b)
# print(rms_rad_vel_b)
# # Optical B
# # Set process noise
# # Qeci = 0 * np.diag([1., 1., 1.])
# # Qric = 5e-14 * np.diag([1., 1., 1.])
# # alpha = 1e-1                            # to tune (in range [1e-4, 1])
# # filter_params = {
# #     'Qeci': Qeci,
# #     'Qric': Qric,
# #     'alpha': alpha,
# #     'gap_seconds': gap_seconds
# # }
# # rms_opt_pos_b, rms_opt_vel_b = analysis(
# #     state_truth_b, state_opt_b, meas_opt_b, sensor_params_opt_b,
# #     int_params, filter_params, t_opt_b, 'optical_b', 'opt', 1)
# # print(rms_opt_pos_b)
# # print(rms_opt_vel_b)

# # OPTIMAL VALUES

# Q = np.asarray([1e-15, 1e-12, 1e-9, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
# n = len(Q)
# rms_rad_pos_a = np.zeros((n, 1))
# rms_rad_vel_a = np.zeros((n, 1))
# rms_opt_pos_a = np.zeros((n, 1))
# rms_opt_vel_a = np.zeros((n, 1))
# rms_rad_pos_b = np.zeros((n, 1))
# rms_rad_vel_b = np.zeros((n, 1))
# rms_opt_pos_b = np.zeros((n, 1))
# rms_opt_vel_b = np.zeros((n, 1))


# def analysis2(truth_state, state, meas,
#              sensor, integ, filters, time):
#     rad2arcsec = 1 / ((1. / 3600.) * np.pi / 180.)
#     filter_output = util.ukf(state, meas, sensor,
#                              integ, filters, None)
#     # extract covars and state
#     covars = np.vstack(
#         [(filter_output[t])['covar'].reshape(1, 36) for t in time])
#     state = np.vstack(
#         [(filter_output[t])['state'].reshape(1, 6) for t in time])
#     dummy = len((filter_output[time[0]])['resids'])
#     resid = np.vstack(
#         [(filter_output[t])['resids'].reshape(1, dummy) for t in time])

#     error_ric = np.zeros((len(time), 3))

#     error = truth_state - state

#     cov_pos_ric = np.zeros((len(time), 9))

#     for i in range(len(time)):
#         # int state
#         error_ric[i, :] = util.eci2ric(
#             state[i, 0:3], state[i, 3:6], error[i, :3]).reshape(1, 3)
#         cov = covars[i, :].reshape(6, 6)
#         cov_pos_ric[i, :] = (util.eci2ric(
#             state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)

#     # sigma bound
#     sigma_bound_pos = 3 * np.sqrt(cov_pos_ric)

#     RMS_pos = np.sqrt(np.sum(error[:, 0:3] ** 2) / len(time))
#     RMS_vel = np.sqrt(np.sum(error[:, 3:6] ** 2) / len(time))

#     return RMS_pos, RMS_vel
# # for i in range(n):
# #     Qeci = Q[i] * np.diag([1., 1., 1.])
# #     Qric = 0 * np.diag([1., 1., 1.])
# #     alpha = 1e-2  # to tune (in range [1e-4, 1])
# #     filter_params = {
# #         'Qeci': Qeci,
# #         'Qric': Qric,
# #         'alpha': alpha,
# #         'gap_seconds': gap_seconds
# #     }
# #     # Radar A
# #     rms_rad_pos_a[i], rms_rad_vel_a[i] = analysis2(
# #         state_truth_a, state_rad_a, meas_rad_a, sensor_params_rad_a,
# #         int_params, filter_params, t_rad_a)
# #     rms_opt_pos_a[i], rms_opt_vel_a[i] = analysis2(
# #         state_truth_a, state_opt_a, meas_opt_a, sensor_params_opt_a,
# #         int_params, filter_params, t_opt_a)
# #     rms_rad_pos_b[i], rms_rad_vel_b[i] = analysis2(
# #         state_truth_b, state_rad_b, meas_rad_b, sensor_params_rad_b,
# #         int_params, filter_params, t_rad_b)
# #     rms_opt_pos_b[i], rms_opt_vel_b[i] = analysis2(
# #         state_truth_b, state_opt_b, meas_opt_b, sensor_params_opt_b,
# #         int_params, filter_params, t_opt_b)
# #
# # fig, (ax1, ax2) = plt.subplots(2, 1)
# # ax1.loglog(Q, rms_rad_pos_a)
# # ax1.loglog(Q, rms_opt_pos_a)
# # ax1.loglog(Q, rms_rad_pos_b)
# # ax1.loglog(Q, rms_opt_pos_b)
# # ax1.set_xlabel('Qeci component [m^2/s^4]')
# # ax1.set_ylabel('3D position RMS [m]')
# # ax1.legend(['Radar A', 'Optical A', 'Radar B', 'Optical B'])
# #
# # ax2.loglog(Q, rms_rad_vel_a)
# # ax2.loglog(Q, rms_opt_vel_a)
# # ax2.loglog(Q, rms_rad_vel_b)
# # ax2.loglog(Q, rms_opt_vel_b)
# # ax2.set_xlabel('Qeci component [m^2/s^4]')
# # ax2.set_ylabel('3D velocity RMS [m/s]')
# # ax2.legend(['Radar A', 'Optical A', 'Radar B', 'Optical B'])
# #
# # plt.tight_layout()
# # plt.savefig('plots/state_noise_compensation.png')