#### Code for Question 3 ####
from pycode.Q3_utilities import *

rso_file = './documentation/Assignment4/group2/q3_meas_iod_99002.pkl'

state_opt, meas_opt, sensor_params_opt = read_measurement_file(
    rso_file)

epochs = meas_opt['tk_list']
sensor_gcrf = np.zeros((len(epochs), 3))
for n_measure in range(len(epochs)):
    eop_alldata = sensor_params_opt['eop_alldata']
    XYs_df = sensor_params_opt['XYs_df']
    UTC = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=epochs[n_measure])
    EOP_data = get_eop_data(eop_alldata, UTC)

    sensor_itrf = sensor_params_opt['sensor_itrf']
    sensor_gcrf_temp, dum = itrf2gcrf(sensor_itrf, np.zeros((3, 1)), UTC, EOP_data, XYs_df)
    sensor_gcrf[n_measure, :] = sensor_gcrf_temp.reshape(3, )

L = np.zeros((len(epochs), 3))  # initialize Line of sight vector

# LOS calculation
for i in range(len(meas_opt['Yk_list'])):
    L[i, :] = line_of_sight(meas_opt['Yk_list'][i][0], meas_opt['Yk_list'][i][1])

plot_earth_and_vectors(sensor_gcrf, L)

# Gauss method ==> Define initial ranges for Gooding method
# rad, ranges = angles_only_gauss(L, epochs[0], epochs[1], epochs[2], sensor_gcrf)

range_initial = 2*(np.sum(np.linalg.norm(sensor_gcrf, axis=1)))
plot_earth_and_ranges(sensor_gcrf, L, range_initial, range_initial)
range1, range2, range3, iterations, lambert_arc_ephemeris, range1_list, range3_list, lambert_arc_ephemeris_initial = (
    gooding_angles_only(L, epochs[0], epochs[1], epochs[2], sensor_gcrf, range_initial, range_initial))
print('Number of iterations Gooding method: ', iterations)
plot_orbit(lambert_arc_ephemeris_initial, lambert_arc_ephemeris, epochs[0], epochs[1], epochs[2], sensor_gcrf, L,
           range_initial, range1, range2, range3)

initial_state = lambert_arc_ephemeris.cartesian_state(epochs[0])

best_state, good_states, std_dev_eci_pos = refine_solution(
    initial_state, epochs, sensor_gcrf, L, sensor_params_opt, range1, range1_list, meas_opt['Yk_list'])

keplerian_elements = element_conversion.cartesian_to_keplerian(initial_state, mu)
keplerian_elements_refined = element_conversion.cartesian_to_keplerian(best_state, mu)
std_dev_eci_vel = np.array([np.std(good_states[:, 3]), np.std(good_states[:, 4]), np.std(good_states[:, 5])])

plot_lambert_and_perturbed(lambert_arc_ephemeris, best_state, epochs[0], epochs[1], epochs[2])

print('######################################')
print('Gooding method solution:')
print('Epochs: ', epochs)
print('Estimated ranges: \nrange 1: ', range1, '\nrange 2: ', range2, '\nrange 3: ', range3)
print('Cartesian State epoch 0: \n', initial_state)
print('Cartesian State epoch 1: \n', lambert_arc_ephemeris.cartesian_state(epochs[1]))
print('Cartesian State epoch 2: \n', lambert_arc_ephemeris.cartesian_state(epochs[2]))
print('Keplerian elements: \n', keplerian_elements[:5])

print('--------------------------------------')
print('Refined solution:')
print('Best initial state: ', best_state)
print('Keplerian elements: ', keplerian_elements_refined[:5])
print('Uncertainty (std dev)')
print('position: ', std_dev_eci_pos)
print('velocity: ', std_dev_eci_vel)

save_as_rso_obj(epochs[0], best_state, std_dev_eci_pos, std_dev_eci_vel)
