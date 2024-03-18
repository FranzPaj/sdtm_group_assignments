import EstimationUtilities as util

# Extract measurements
state_opt_a, meas_opt_a, sensor_params_opt_a = util.read_measurement_file(
    './Data/group2/group2_q2a_skymuster_sparse_optical_meas.pkl')

state_rad_a, meas_rad_a, sensor_params_rad_a = util.read_measurement_file(
    './Data/group2/group2_q2a_skymuster_sparse_radar_meas.pkl')

state_opt_b, meas_opt_b, sensor_params_opt_b = util.read_measurement_file(
    './Data/group2/group2_q2b_skymuster_sparse_optical_meas.pkl')

state_rad_b, meas_rad_b, sensor_params_rad_b = util.read_measurement_file(
    './Data/group2/group2_q2b_skymuster_sparse_radar_meas.pkl')

t_truth_a, state_truth_a, params_truth_a = util.read_truth_file(
    './Data/group2/group2_q2a_skymuster_sparse_truth_grav.pkl')

t_truth_b, state_truth_b, params_truth_b = util.read_truth_file(
    './Data/group2/group2_q2b_skymuster_sparse_truth_grav_srp.pkl')

################################# POINT A #####################################

# ukf(state_params, meas_dict, sensor_params,
# int_params, filter_params, bodies)
# where filter_params: Q_eci and Q_ric

# then we need to find values for noise in ECI and RIC frame


