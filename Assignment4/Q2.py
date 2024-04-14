import datetime

import tudatpy.astro.two_body_dynamics
from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup

import pycode.EstimationUtilities as est_util
import pycode.TudatPropagator as prop
import numpy as np
from tudatpy.astro import element_conversion, two_body_dynamics
import matplotlib.pyplot as plt
import pickle
from tudatpy.astro import two_body_dynamics

###############################################################################
##################### LAMBERT TARGETER FUNCTIONS ##############################
###############################################################################


def get_lambert_result(
        departure_epoch,
        initial_state_,
        arrival_epoch,
        final_state_):

    # Gravitational parameter of the Sun
    mu_earth = spice.get_body_gravitational_parameter('Earth')

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state_[:3], final_state_[:3],
        arrival_epoch - departure_epoch,
        mu_earth)

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state_
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    return lambert_arc_initial_state


###############################################################################

def julian2calendar(jd):
    from math import trunc
    jd0 = jd + 0.5
    l1 = trunc(jd0 + 68569)
    l2 = trunc(4 * l1 / 146097)
    l3 = l1 - trunc((146097 * l2 + 3) / 4)
    l4 = trunc(4000 * (l3 + 1) / 1461001)
    l5 = l3 - trunc(1461 * l4 / 4) + 31
    l6 = trunc(80 * l5 / 2447)
    l7 = trunc(l6 / 11)
    d = l5 - trunc(2447 * l6 / 80)
    m = l6 + 2 - 12 * l7
    y = 100 * (l2 - 49) + l4 + l7
    # computing minutes and seconds
    fr = jd % 1 - 0.5
    if fr < 0:
        fr = 1 + fr
    h = trunc(fr * 24)

    rest = fr * 86400 - h * 3600
    p = trunc(rest / 60)
    s = round(rest - p * 60)
    cdate = [y, m, d, h, p, s]
    return cdate

###############################################################################
###############################################################################


############################### QUESTION 2 ####################################
# Constants
mu_e = spice.get_body_gravitational_parameter('Earth')
rad2arcsec = 1 / ((1. / 3600.) * np.pi / 180.)
j2000 = 2451545 * 86400
# Extract measurements
state_opt, meas_opt, sensor_params_opt = est_util.read_measurement_file(
    'data/states_updated/q2_meas_maneuver_91686.pkl')

initial_state = state_opt['state']
initial_covar = state_opt['covar']

# Window times
t_opt = np.asarray(meas_opt['tk_list'])
gaps = np.diff(t_opt)
gap_opt = np.max(np.diff(t_opt))
initial_time = state_opt['UTC']

# Split data into windows
flags = [0]
for i in range(len(gaps)):
    if gaps[i] > 10:
        flags.append(i+1)
flags.append(-1)
data_split = {}
state_initial = state_opt
# Settings
integ = {'tudat_integrator': 'rkf78',
         'step': 10.,
         'max_step': 1000.,
         'min_step': 1e-3,
         'rtol': 1e-12,
         'atol': 1e-12}
Qeci = 0 * np.diag([1., 1., 1.])
Qric = 1e-12 * np.diag([1., 1., 1.])
alpha = 1e-1
filters = {
    'Qeci': Qeci,
    'Qric': Qric,
    'alpha': alpha,
    'gap_seconds': 100
}
if 0:
    filter_output = est_util.ukf(state_initial, meas_opt, sensor_params_opt,
                                 integ, filters, None)
    # SAVE DATA
    with open('output/ukf_global.pkl', 'wb') as f:
        pickle.dump(filter_output, f)


# DATA PROCESSING
with open('output/ukf_global.pkl', 'rb') as f:
    ukf_output = pickle.load(f)
# extract covars and state
covars0 = np.vstack(
    [(ukf_output[t])['covar'].reshape(1, 36) for t in t_opt])
state0 = np.vstack(
    [(ukf_output[t])['state'].reshape(1, 6) for t in t_opt])
resid0 = np.vstack(
    [(ukf_output[t])['resids'].reshape(
        1, len((ukf_output[t_opt[0]])['resids'])) for t in t_opt])
# Plotting
t_relative = (t_opt - t_opt[0]) / 3600
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(t_relative, resid0[:, 0] * rad2arcsec)
ax1.set_ylabel('RA res. [arcsec]', size=14)
ax1.grid()
ax2.scatter(t_relative, resid0[:, 1] * rad2arcsec)
ax2.set_xlabel('Relative time [hours]', size=14)
ax2.set_ylabel('DEC res. [arcsec]', size=14)
ax2.grid()
plt.savefig('output/all_windows.png')
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(t_relative[180:240], resid0[180:240, 0] * rad2arcsec)
ax1.set_ylabel('RA res. [arcsec]', size=14)
ax1.grid()
ax2.scatter(t_relative[180:240], resid0[180:240, 1] * rad2arcsec)
ax2.set_xlabel('Relative time [hours]', size=14)
ax2.set_ylabel('DEC res. [arcsec]', size=14)
ax2.grid()
plt.savefig('output/window4.png')
# window selected is number 4
resid_w4 = resid0[180:240, :]
t_w4 = t_opt[180:240]
state_w4 = state0[180:240]
flag = True
i = 9
position = 0
std_w4 = np.std(resid_w4[0:i, 0])
while flag is True:
    if np.abs(resid_w4[i+1, 0]) > 3*std_w4:
        flag = False
        position = i
    i += 1
    std_w4 = np.std(resid_w4[0:i])

t_estimate_before = t_w4[position]
t_estimate_before_rel = t_w4[position] - t_opt[0]
covars_before = covars0[180:240, :]
t_before_utc = julian2calendar((t_estimate_before + j2000)/86400)
print(t_estimate_before, t_estimate_before_rel, t_before_utc)
t_estimate_after = t_w4[position + 1]
t_estimate_after_rel = t_w4[position + 1] - t_opt[0]
t_after_utc = julian2calendar((t_estimate_after + j2000)/86400)
print(t_estimate_after, t_estimate_after_rel, t_after_utc)
state_before = state_w4[position, :]
state_after = state_w4[position+1, :]

# RESTART UKF after maneuver
t_after = t_opt[180+position+1:-1]
delta = t_after[0] - t_opt[0]
state_initial['UTC'] = datetime.datetime(2024, 3, 21, 16, 12, 9)
state_initial['state'] = state_after.reshape(6, 1)
state_initial['covar'] = np.array([[1e6, 1e4, 1e4, 0, 0, 0],
                                   [1e4, 1e6, 1e4, 0, 0, 0],
                                   [1e4, 1e4, 1e6, 0, 0, 0],
                                   [0, 0, 0, 1e4, 1e3, 1e3],
                                   [0, 0, 0, 1e3, 1e4, 1e3],
                                   [0, 0, 0, 1e3, 1e3, 1e4]])
meas = {'tk_list': list(t_after),
        'Yk_list': (meas_opt['Yk_list'])[180+position+1:-1]}
if 0:
    after_man = est_util.ukf(state_initial, meas, sensor_params_opt,
                                 integ, filters, None)
    # SAVE DATA
    with open('output/ukf_afterman.pkl', 'wb') as f:
        pickle.dump(after_man, f)

# extract covars and state
with open('output/ukf_afterman.pkl', 'rb') as f:
    after_man = pickle.load(f)
covars = np.vstack(
    [(after_man[t])['covar'].reshape(1, 36) for t in t_after])
state = np.vstack(
    [(after_man[t])['state'].reshape(1, 6) for t in t_after])
resid = np.vstack(
    [(after_man[t])['resids'].reshape(
        1, len((after_man[t_after[0]])['resids'])) for t in t_after])
t_after_relative = (t_after - t_opt[0]) / 3600
# plot all windows after maneuver corrected
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(t_after_relative[1:], resid[1:, 0] * rad2arcsec)
ax1.set_ylabel('RA res. [arcsec]', size=14)
ax1.grid()
ax2.scatter(t_after_relative[1:], resid[1:, 1] * rad2arcsec)
ax2.set_xlabel('Relative time [hours]', size=14)
ax2.set_ylabel('DEC res. [arcsec]', size=14)
ax2.grid()
plt.savefig('output/all_windows_after_man.png')
# plot window 4 after maneuver
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(t_after_relative[1:240-(181+position)],
            resid[1:240-(181+position), 0] * rad2arcsec)
ax1.set_ylabel('RA res. [arcsec]', size=14)
ax1.grid()
ax2.scatter(t_after_relative[1:240-(181+position)],
            resid[1:240-(181+position), 1] * rad2arcsec)
ax2.set_xlabel('Relative time [hours]', size=14)
ax2.set_ylabel('DEC res. [arcsec]', size=14)
ax2.grid()
plt.savefig('output/window4_after_man.png')

# NUMERICAL PROPAGATION TO FIND INTERSECTION
# forward propagation from  point before maneuver
x_before = state_before.reshape(6, 1)
cov_before = covars_before[position, :].reshape(6, 6)
t_before = t_estimate_before
integ_before = {'tudat_integrator': 'rkf78',
                'step': 1e-1,
                'max_step': 1e-1,
                'min_step': 1e-1,
                'rtol': 1e-12,
                'atol': 1e-12}
time_b, x_b, P_b = prop.propagate_state_and_covar_q2(
    x_before, cov_before, [t_before, t_estimate_after],
    state_opt, integ_before)

# Apply Lambert Arc on final states of window 4 to find a good starting state
t_w4_after_man = t_after[0:242-(181+position)]
state_w4_after_man = state[0:242-(181+position), :]
covars_w4_after_man = covars[0:242-(181+position), :]
# last state and time of window 4
last_state_w4 = state_w4_after_man[-1, :]
last_time_w4 = t_w4_after_man[-1]
# other point to define lambert arc
lamb_state_w4 = state_w4_after_man[-5, :]
lamb_time_w4 = t_w4_after_man[-5]
# get v-corrected state using lambert arc
vcorrected_state = get_lambert_result(lamb_time_w4, lamb_state_w4,
                                      last_time_w4, last_state_w4)

x_after = vcorrected_state.reshape(6, 1)
t_a = lamb_time_w4
cov_after = covars_w4_after_man[-5, :].reshape(6, 6)
integ_after = {'tudat_integrator': 'rkf78',
               'step': -1e-1,
               'max_step': -1e-1,
               'min_step': -1e-1,
               'rtol': 1e-12,
               'atol': 1e-12}
time_a, x_a, P_a = prop.propagate_state_and_covar_q2(x_after, cov_after,
                                                     [t_a, t_before],
                                                     state_opt, integ_after)

# find maneuver point
diff = np.linalg.norm(x_b - x_a[:len(time_b), :], axis=1)
flag_man = np.argmin(diff)

time_man = initial_time + datetime.timedelta(
    seconds=time_b[flag_man] - t_opt[0])
print(time_man)
delta_pos = np.linalg.norm(x_a[flag_man, 0:3] - x_b[flag_man, 0:3])
print(delta_pos)
delta_v_vector = x_a[flag_man, 3:6] - x_b[flag_man, 3:6]
delta_v = np.linalg.norm(delta_v_vector)
print(delta_v)
print(delta_v_vector)

# Uncertainty of maneuver point
P_man_a = P_a[flag_man, :].reshape(6, 6)
print(P_man_a[3:, 3:])
P_man_b = P_b[flag_man, :].reshape(6, 6)
print(P_man_b[3:, 3:])
sigma_vel = np.sqrt(P_man_a[3, 3]+P_man_a[4, 4]+P_man_a[5, 5])
sigma_pos = np.sqrt(P_man_a[0, 0]+P_man_a[1, 1]+P_man_a[2, 2])
print(sigma_vel)
print(sigma_pos)
# Dimension check
r_b = np.linalg.norm(x_b[:, :3], axis=1)
r_a = np.linalg.norm(x_a[:, :3], axis=1)
print(r_a)
print(r_b)
# states after man from UKF
r_after = np.linalg.norm(state, axis=1)
print(r_after)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x_b[:, 0], x_b[:, 1], x_b[:, 2])
# ax.plot3D(x_a[:, 0], x_a[:, 1], x_a[:, 2])
# ax.plot3D(state0[:, 0], state0[:, 1], state0[:, 2])
# plt.show()

# Convert to keplerian elements
state_kep_a = np.zeros((len(t_after), 6))
for j in range(len(t_after)):
    state_kep_a[j, :] = element_conversion.cartesian_to_keplerian(
        state[j, :], mu_e)
dummy_a = np.average(state_kep_a, axis=0)
print(dummy_a)
print(np.std(state_kep_a, axis=0))
# Convert to keplerian elements
state_kep_b = np.zeros((len(t_opt[:180+position]), 6))
for j in range(len(t_opt[:180+position])):
    state_kep_b[j, :] = element_conversion.cartesian_to_keplerian(
        state0[j, :], mu_e)
dummy_b = np.average(state_kep_b, axis=0)
print(dummy_b)
print(np.std(state_kep_b, axis=0))

# COMPUTE RMS of residuals
rms_preman = np.sqrt(np.sum((resid0[0:181+position]*rad2arcsec)**2)/
                     len(resid0[0:181+position]))
rms_afterman = np.sqrt(np.sum((resid*rad2arcsec)**2)/len(resid))
print(rms_preman, rms_afterman)

# COMPUTE sigma bounds
time_before = t_opt[0:180+position]
time_after = t_opt[180+position:]
covariance_before = covars0[0:180+position, :]
covariance_after = covars
ric_before = np.zeros((len(time_before), 9))
ric_after = np.zeros((len(time_after)-2, 9))

for i in range(len(time_before)):
    cov = covariance_before[i, :].reshape(6, 6)
    ric_before[i, :] = (est_util.eci2ric(
        state0[i, 0:3], state0[i, 3:6], cov[:3, :3])).reshape(1, 9)
r_before = 3*np.sqrt(ric_before[:, 0])
i_before = 3*np.sqrt(ric_before[:, 4])
c_before = 3*np.sqrt(ric_before[:, 8])
for i in range(len(time_after)-2):
    cov = covariance_after[i, :].reshape(6, 6)
    ric_after[i, :] = (est_util.eci2ric(
        state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)
r_after = 3*np.sqrt(ric_after[:, 0])
i_after = 3*np.sqrt(ric_after[:, 4])
c_after = 3*np.sqrt(ric_after[:, 8])
t_opt = (t_opt - t_opt[0])/3600
# plot sigma bounds
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = (
    plt.subplots(3, 4, figsize=(8, 6)))
# w1
ax1.plot(t_opt[0:60], r_before[0:60], label='R-axis')
ax5.plot(t_opt[0:60], i_before[0:60], label='I-axis')
ax9.plot(t_opt[0:60], c_before[0:60], label='C-axis')
ax1.plot(t_opt[0:60], -r_before[0:60], label='R-axis')
ax5.plot(t_opt[0:60], -i_before[0:60], label='I-axis')
ax9.plot(t_opt[0:60], -c_before[0:60], label='C-axis')
# w2
ax2.plot(t_opt[60:120], r_before[60:120], label='R component')
ax6.plot(t_opt[60:120], i_before[60:120], label='I component')
ax10.plot(t_opt[60:120], c_before[60:120], label='C component')
ax2.plot(t_opt[60:120], -r_before[60:120], label='R component')
ax6.plot(t_opt[60:120], -i_before[60:120], label='I component')
ax10.plot(t_opt[60:120], -c_before[60:120], label='C component')
# w3
ax3.plot(t_opt[120:180], r_before[120:180], label='R component')
ax7.plot(t_opt[120:180], i_before[120:180], label='I component')
ax11.plot(t_opt[120:180], c_before[120:180], label='C component')
ax3.plot(t_opt[120:180], -r_before[120:180], label='R component')
ax7.plot(t_opt[120:180], -i_before[120:180], label='I component')
ax11.plot(t_opt[120:180], -c_before[120:180], label='C component')
# w4
ax4.plot(t_opt[180:180+position], r_before[180:], label='R component')
ax8.plot(t_opt[180:180+position], i_before[180:], label='I component')
ax12.plot(t_opt[180:180+position], c_before[180:], label='C component')
ax4.plot(t_opt[180:180+position], -r_before[180:], label='R component')
ax8.plot(t_opt[180:180+position], -i_before[180:], label='I component')
ax12.plot(t_opt[180:180+position], -c_before[180:], label='C component')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()
ax9.grid()
ax10.grid()
ax11.grid()
ax12.grid()

ax1.set_ylabel('R-axis [m]', size=15)
ax5.set_ylabel('I-axis [m]', size=15)
ax9.set_ylabel('C-axis [m]', size=15)
fig.text(0.5, 0.0, 'Relative time [hours]', ha='center', size=15)
plt.tight_layout()
plt.savefig('output/sbound_before.png')

# plot sigma bounds
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = (
    plt.subplots(3, 4, figsize=(8, 6)))
# w1
ax1.plot(t_opt[181+position:240], r_after[0:240-(181+position)], label='R-axis')
ax5.plot(t_opt[181+position:240], i_after[0:240-(181+position)], label='I-axis')
ax9.plot(t_opt[181+position:240], c_after[0:240-(181+position)], label='C-axis')
ax1.plot(t_opt[181+position:240], -r_after[0:240-(181+position)], label='R-axis')
ax5.plot(t_opt[181+position:240], -i_after[0:240-(181+position)], label='I-axis')
ax9.plot(t_opt[181+position:240], -c_after[0:240-(181+position)], label='C-axis')
# w2
ax2.plot(t_opt[240:300], r_after[-179:-119], label='R component')
ax6.plot(t_opt[240:300], i_after[-179:-119], label='I component')
ax10.plot(t_opt[240:300], c_after[-179:-119], label='C component')
ax2.plot(t_opt[240:300], -r_after[-179:-119], label='R component')
ax6.plot(t_opt[240:300], -i_after[-179:-119], label='I component')
ax10.plot(t_opt[240:300], -c_after[-179:-119], label='C component')
# w3
ax3.plot(t_opt[300:360], r_after[-119:-59], label='R component')
ax7.plot(t_opt[300:360], i_after[-119:-59], label='I component')
ax11.plot(t_opt[300:360], c_after[-119:-59], label='C component')
ax3.plot(t_opt[300:360], -r_after[-119:-59], label='R component')
ax7.plot(t_opt[300:360], -i_after[-119:-59], label='I component')
ax11.plot(t_opt[300:360], -c_after[-119:-59], label='C component')
# w4
ax4.plot(t_opt[360:419], r_after[-59:], label='R component')
ax8.plot(t_opt[360:419], i_after[-59:], label='I component')
ax12.plot(t_opt[360:419], c_after[-59:], label='C component')
ax4.plot(t_opt[360:419], -r_after[-59:], label='R component')
ax8.plot(t_opt[360:419], -i_after[-59:], label='I component')
ax12.plot(t_opt[360:419], -c_after[-59:], label='C component')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()
ax9.grid()
ax10.grid()
ax11.grid()
ax12.grid()

ax1.set_ylabel('R-axis [m]', size=15)
ax5.set_ylabel('I-axis [m]', size=15)
ax9.set_ylabel('C-axis [m]', size=15)
fig.text(0.5, 0.0, 'Relative time [hours]', ha='center', size=15)
plt.tight_layout()
plt.savefig('output/sbound_after.png')


# PROPAGATE KEPLERIAN ORBITS
# time_keplerian = np.arange(0, 120000, 100)
# states_car_a = np.zeros((len(time_keplerian), 6))
# states_car_b = np.zeros((len(time_keplerian), 6))
# for i in range(len(time_keplerian)):
#     states_car_a[i, :] = element_conversion.keplerian_to_cartesian(
#         two_body_dynamics.propagate_kepler_orbit(
#         dummy_a, time_keplerian[i], mu_e), mu_e)
#     states_car_b[i, :] = element_conversion.keplerian_to_cartesian(
#         two_body_dynamics.propagate_kepler_orbit(
#         dummy_b, time_keplerian[i], mu_e), mu_e)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(states_car_a[:, 0], states_car_a[:, 1], states_car_a[:, 2],
#           label='After maneuver')
# ax.plot3D(states_car_b[:, 0], states_car_b[:, 1], states_car_b[:, 2],
#           label='Before Maneuver')
# ax.set_xlabel('x [m]', size=14)
# ax.set_ylabel('y [m]', size=14)
# ax.set_zlabel('z [m]', size=14, labelpad=8, )
# plt.legend()
# ax.axis('equal')
# ax.view_init(elev=45, azim=45)
# plt.savefig('output/orbits.png')
