import EstimationUtilities as util
import numpy as np
import matplotlib.pyplot as plt
import math
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

bodies = state_opt_a['bodies_to_create']
################################# POINT A #####################################
# Develop a procedure to adjust the SNC covariance to process the measurements
# using the UKF. State your final selection with justification. You must
# include, at minimum, a plot of your position errors with 3-σ covariance
# bounds in the Radial-Intrack-Crosstrack (RIC) coordinate frame, as well as
# a plot of post-fit measurement residuals using your selected SNC parameters.
# You may include any additional plots or tables at your discretion.


# ukf(state_params, meas_dict, sensor_params,
# int_params, filter_params, bodies)
# where filter_params: Q_eci and Q_ric


# Tune biggest gap in seconds between measurements
t_opt_a = np.asarray(meas_opt_a['tk_list'])
t_opt_b = np.asarray(meas_opt_b['tk_list'])
t_rad_a = np.asarray(meas_rad_a['tk_list'])
t_rad_b = np.asarray(meas_rad_b['tk_list'])
time_gaps = np.diff(t_opt_a)
gap_truth_a = np.max(np.diff(t_truth_a))
gap_truth_b = np.max(np.diff(t_truth_b))
gap_opt_a = np.max(np.diff(t_opt_a))
gap_opt_b = np.max(np.diff(t_opt_b))
gap_rad_a = np.max(np.diff(t_rad_a))
gap_rad_b = np.max(np.diff(t_rad_b))

gap_seconds = 6*3600                  # to tune

int_params = {'tudat_integrator': 'rkf78',
              'step': 10.,
              'max_step': 1000.,
              'min_step': 1e-3,
              'rtol': 1e-12,
              'atol': 1e-12}


def analysis(truth_state, state, meas,
             sensor, integ, filters, time, name: str, mtype, plotting):
    rad2arcsec = 1 / ((1. / 3600.) * np.pi / 180.)
    filter_output = util.ukf(state, meas, sensor,
                             integ, filters, None)
    # extract covars and state
    covars = np.vstack(
        [(filter_output[t])['covar'].reshape(1, 36) for t in time])
    state = np.vstack(
        [(filter_output[t])['state'].reshape(1, 6) for t in time])
    dummy = len((filter_output[time[0]])['resids'])
    resid = np.vstack(
        [(filter_output[t])['resids'].reshape(1, dummy) for t in time])

    error_ric = np.zeros((len(time), 3))

    error = truth_state - state

    cov_pos_ric = np.zeros((len(time), 9))

    for i in range(len(time)):
        # int state
        error_ric[i, :] = util.eci2ric(
            state[i, 0:3], state[i, 3:6], error[i, :3]).reshape(1, 3)
        cov = covars[i, :].reshape(6, 6)
        cov_pos_ric[i, :] = (util.eci2ric(
            state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)

    # sigma bound
    sigma_bound_pos = 3 * np.sqrt(cov_pos_ric)

    RMS_pos = np.sqrt(np.sum(error[:, 0:3]**2)/len(time))
    RMS_vel = np.sqrt(np.sum(error[:, 3:6]**2)/len(time))
    if mtype == 'opt':
        RMS_RA = np.sqrt(np.sum((resid[:, 0]*rad2arcsec) ** 2) / len(time))
        RMS_DEC = np.sqrt(np.sum((resid[:, 1]*rad2arcsec) ** 2) / len(time))
        print(RMS_RA)
        print(RMS_DEC)
    if mtype == 'rad':
        RMS_range = np.sqrt(np.sum(resid[:, 0] ** 2) / len(time))
        RMS_RA = np.sqrt(np.sum((resid[:, 1]*rad2arcsec) ** 2) / len(time))
        RMS_DEC = np.sqrt(np.sum((resid[:, 2]*rad2arcsec) ** 2) / len(time))
        print(RMS_range)
        print(RMS_RA)
        print(RMS_DEC)

    if plotting:
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        # plotting 1
        time_rel = (time - time[0]) / 3600
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(time_rel, error_ric[:, 0])
        ax1.plot(time_rel, sigma_bound_pos[:, 0], 'k:')
        ax1.plot(time_rel, -sigma_bound_pos[:, 0], 'k:')
        ax1.grid()
        ax2.plot(time_rel, error_ric[:, 1])
        ax2.plot(time_rel, sigma_bound_pos[:, 4], 'k:')
        ax2.plot(time_rel, -sigma_bound_pos[:, 4], 'k:')
        ax2.grid()
        ax3.plot(time_rel, error_ric[:, 2])
        ax3.plot(time_rel, sigma_bound_pos[:, 8], 'k:')
        ax3.plot(time_rel, -sigma_bound_pos[:, 8], 'k:')
        ax3.grid()
        plt.tight_layout()
        plt.savefig('plots/' + name + '_error.png')

        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
              (ax9, ax10, ax11, ax12)) = plt.subplots(
            3, 4, figsize=(12, 8))
        # R direction
        ax1.plot(time_rel[:31], error_ric[:31, 0])
        ax1.plot(time_rel[:31], sigma_bound_pos[:31, 0], 'k:')
        ax1.plot(time_rel[:31], -sigma_bound_pos[:31, 0], 'k:')
        ax1.grid()
        ax1.set_ylabel('R-direction [m]', size=18)
        ax2.plot(time_rel[31:62], error_ric[31:62, 0])
        ax2.plot(time_rel[31:62], sigma_bound_pos[31:62, 0], 'k:')
        ax2.plot(time_rel[31:62], -sigma_bound_pos[31:62, 0], 'k:')
        ax2.grid()
        ax3.plot(time_rel[62:93], error_ric[62:93, 0])
        ax3.plot(time_rel[62:93], sigma_bound_pos[62:93, 0], 'k:')
        ax3.plot(time_rel[62:93], -sigma_bound_pos[62:93, 0], 'k:')
        ax3.grid()
        ax4.plot(time_rel[93:], error_ric[93:, 0])
        ax4.plot(time_rel[93:], sigma_bound_pos[93:, 0], 'k:')
        ax4.plot(time_rel[93:], -sigma_bound_pos[93:, 0], 'k:')
        ax4.grid()
        # I direction
        ax5.plot(time_rel[:31], error_ric[:31, 1])
        ax5.set_ylabel('I-direction [m]', size=18)
        ax5.plot(time_rel[:31], sigma_bound_pos[:31, 4], 'k:')
        ax5.plot(time_rel[:31], -sigma_bound_pos[:31, 4], 'k:')
        ax5.grid()
        ax6.plot(time_rel[31:62], error_ric[31:62, 1])
        ax6.plot(time_rel[31:62], sigma_bound_pos[31:62, 4], 'k:')
        ax6.plot(time_rel[31:62], -sigma_bound_pos[31:62, 4], 'k:')
        ax6.grid()
        ax7.plot(time_rel[62:93], error_ric[62:93, 1])
        ax7.plot(time_rel[62:93], sigma_bound_pos[62:93, 4], 'k:')
        ax7.plot(time_rel[62:93], -sigma_bound_pos[62:93, 4], 'k:')
        ax7.grid()
        ax8.plot(time_rel[93:], error_ric[93:, 1])
        ax8.plot(time_rel[93:], sigma_bound_pos[93:, 4], 'k:')
        ax8.plot(time_rel[93:], -sigma_bound_pos[93:, 4], 'k:')
        ax8.grid()
        # C direction
        ax9.plot(time_rel[:31], error_ric[:31, 2])
        ax9.plot(time_rel[:31], sigma_bound_pos[:31, 8], 'k:')
        ax9.plot(time_rel[:31], -sigma_bound_pos[:31, 8], 'k:')
        ax9.set_ylabel('C-direction [m]', size=18)
        ax9.set_xlabel('Time [hours]', size=18)
        ax9.grid()
        ax10.plot(time_rel[31:62], error_ric[31:62, 2])
        ax10.plot(time_rel[31:62], sigma_bound_pos[31:62, 8], 'k:')
        ax10.plot(time_rel[31:62], -sigma_bound_pos[31:62, 8], 'k:')
        ax10.set_xlabel('Time [hours]', size=18)
        ax10.grid()
        ax11.plot(time_rel[62:93], error_ric[62:93, 2])
        ax11.plot(time_rel[62:93], sigma_bound_pos[62:93, 8], 'k:')
        ax11.plot(time_rel[62:93], -sigma_bound_pos[62:93, 8], 'k:')
        ax11.set_xlabel('Time [hours]', size=18)
        ax11.grid()
        ax12.plot(time_rel[93:], error_ric[93:, 2])
        ax12.plot(time_rel[93:], sigma_bound_pos[93:, 8], 'k:')
        ax12.plot(time_rel[93:], -sigma_bound_pos[93:, 8], 'k:')
        ax12.set_xlabel('Time [hours]', size=18)
        ax12.grid()
        plt.tight_layout()
        plt.savefig('plots/' + name + '_error_sparse.png')

        if mtype == 'rad':

            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
                (ax9, ax10, ax11, ax12)) = plt.subplots(
                3, 4, figsize=(12, 8))
            # R direction
            ax1.scatter(time_rel[:31], resid[:31, 0])
            ax1.grid()
            ax1.set_ylabel('Range [m]', size=18)
            ax2.scatter(time_rel[31:62], resid[31:62, 0])
            ax2.grid()
            ax3.scatter(time_rel[62:93], resid[62:93, 0])
            ax3.grid()
            ax4.scatter(time_rel[93:], resid[93:, 0])
            ax4.grid()
            # I direction
            ax5.scatter(time_rel[:31], resid[:31, 1]*rad2arcsec)
            ax5.set_ylabel('RA [arcsec]', size=18)
            ax5.grid()
            ax6.scatter(time_rel[31:62], resid[31:62, 1]*rad2arcsec)
            ax6.grid()
            ax7.scatter(time_rel[62:93], resid[62:93, 1]*rad2arcsec)
            ax7.grid()
            ax8.scatter(time_rel[93:], resid[93:, 1]*rad2arcsec)
            ax8.grid()
            # C direction
            ax9.scatter(time_rel[:31], resid[:31, 2]*rad2arcsec)
            ax9.set_ylabel('DEC [arcsec]', size=18)
            ax9.set_xlabel('Time [hours]', size=18)
            ax9.grid()
            ax10.scatter(time_rel[31:62], resid[31:62, 2]*rad2arcsec)
            ax10.set_xlabel('Time [hours]', size=18)
            ax10.grid()
            ax11.scatter(time_rel[62:93], resid[62:93, 2]*rad2arcsec)
            ax11.set_xlabel('Time [hours]', size=18)
            ax11.grid()
            ax12.scatter(time_rel[93:], resid[93:, 2]*rad2arcsec)
            ax12.set_xlabel('Time [hours]', size=18)
            ax12.grid()
            plt.tight_layout()
            plt.savefig('plots/' + name + '_residuals.png')

        elif mtype == 'opt':
            rad2arcsec = 3600. * 180. / np.pi
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
                2, 4, figsize=(16, 10))
            # R direction
            ax1.scatter(time_rel[:31], resid[:31, 0] * rad2arcsec )
            ax1.set_ylabel('RA [arcsec]', size=18)
            ax1.grid()
            ax2.scatter(time_rel[31:62], resid[31:62, 0] * rad2arcsec)
            ax2.grid()
            ax3.scatter(time_rel[62:93], resid[62:93, 0] * rad2arcsec)
            ax3.grid()
            ax4.scatter(time_rel[93:], resid[93:, 0] * rad2arcsec)
            ax4.grid()
            # I direction
            ax5.scatter(time_rel[:31], resid[:31, 1] * rad2arcsec)
            ax5.set_ylabel('DEC [arcsec]', size=18)
            ax5.set_xlabel('Time [hours]', size=18)
            ax5.grid()
            ax6.scatter(time_rel[31:62], resid[31:62, 1] * rad2arcsec)
            ax6.set_xlabel('Time [hours]', size=18)
            ax6.grid()
            ax7.scatter(time_rel[62:93], resid[62:93, 1] * rad2arcsec)
            ax7.set_xlabel('Time [hours]', size=18)
            ax7.grid()
            ax8.scatter(time_rel[93:], resid[93:, 1] * rad2arcsec)
            ax8.set_xlabel('Time [hours]', size=18)
            ax8.grid()
            plt.tight_layout()
            plt.savefig('plots/' + name + '_residuals.png')

    return RMS_pos, RMS_vel


# Radar A
# Set process noise
# Qeci = 0 * np.diag([1., 1., 1.])
# Qric = 8e-15 * np.diag([1., 1., 1.])
# alpha = 1e-4                            # to tune (in range [1e-4, 1])
# filter_params = {
#     'Qeci': Qeci,
#     'Qric': Qric,
#     'alpha': alpha,
#     'gap_seconds': gap_seconds
# }
# rms_rad_pos_a, rms_rad_vel_a = analysis(
#     state_truth_a, state_rad_a, meas_rad_a, sensor_params_rad_a,
#     int_params, filter_params, t_rad_a, 'radar_a', 'rad', 1)
# print(rms_rad_pos_a)
# print(rms_rad_vel_a)
# # Optical A
# Set process noise
# Qeci = 1e-9 * np.diag([1., 1., 1.])
# Qric = 0 * np.diag([20., 1., 1.])
# alpha = 1e-4                            # to tune (in range [1e-4, 1])
# filter_params = {
#     'Qeci': Qeci,
#     'Qric': Qric,
#     'alpha': alpha,
#     'gap_seconds': gap_seconds
# }
# rms_opt_pos_a, rms_opt_vel_a = analysis(
#     state_truth_a, state_opt_a, meas_opt_a, sensor_params_opt_a,
#     int_params, filter_params, t_opt_a, 'optical_a', 'opt', 1)
# print(rms_opt_pos_a)
# print(rms_opt_vel_a)

# Radar B
# Set process noise
Qeci = 0 * np.diag([1., 1., 1.])
Qric = 8e-12 * np.diag([1., 1., 1.])
alpha = 1e-4                            # to tune (in range [1e-4, 1])
filter_params = {
    'Qeci': Qeci,
    'Qric': Qric,
    'alpha': alpha,
    'gap_seconds': gap_seconds
}
rms_rad_pos_b, rms_rad_vel_b = analysis(
    state_truth_b, state_rad_b, meas_rad_b, sensor_params_rad_b,
    int_params, filter_params, t_rad_b, 'radar_b', 'rad', 1)
print(rms_rad_pos_b)
print(rms_rad_vel_b)
# Optical B
# Set process noise
# Qeci = 0 * np.diag([1., 1., 1.])
# Qric = 5e-14 * np.diag([1., 1., 1.])
# alpha = 1e-1                            # to tune (in range [1e-4, 1])
# filter_params = {
#     'Qeci': Qeci,
#     'Qric': Qric,
#     'alpha': alpha,
#     'gap_seconds': gap_seconds
# }
# rms_opt_pos_b, rms_opt_vel_b = analysis(
#     state_truth_b, state_opt_b, meas_opt_b, sensor_params_opt_b,
#     int_params, filter_params, t_opt_b, 'optical_b', 'opt', 1)
# print(rms_opt_pos_b)
# print(rms_opt_vel_b)

# OPTIMAL VALUES

Q = np.asarray([1e-15, 1e-12, 1e-9, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
n = len(Q)
rms_rad_pos_a = np.zeros((n, 1))
rms_rad_vel_a = np.zeros((n, 1))
rms_opt_pos_a = np.zeros((n, 1))
rms_opt_vel_a = np.zeros((n, 1))
rms_rad_pos_b = np.zeros((n, 1))
rms_rad_vel_b = np.zeros((n, 1))
rms_opt_pos_b = np.zeros((n, 1))
rms_opt_vel_b = np.zeros((n, 1))


def analysis2(truth_state, state, meas,
             sensor, integ, filters, time):
    rad2arcsec = 1 / ((1. / 3600.) * np.pi / 180.)
    filter_output = util.ukf(state, meas, sensor,
                             integ, filters, None)
    # extract covars and state
    covars = np.vstack(
        [(filter_output[t])['covar'].reshape(1, 36) for t in time])
    state = np.vstack(
        [(filter_output[t])['state'].reshape(1, 6) for t in time])
    dummy = len((filter_output[time[0]])['resids'])
    resid = np.vstack(
        [(filter_output[t])['resids'].reshape(1, dummy) for t in time])

    error_ric = np.zeros((len(time), 3))

    error = truth_state - state

    cov_pos_ric = np.zeros((len(time), 9))

    for i in range(len(time)):
        # int state
        error_ric[i, :] = util.eci2ric(
            state[i, 0:3], state[i, 3:6], error[i, :3]).reshape(1, 3)
        cov = covars[i, :].reshape(6, 6)
        cov_pos_ric[i, :] = (util.eci2ric(
            state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)

    # sigma bound
    sigma_bound_pos = 3 * np.sqrt(cov_pos_ric)

    RMS_pos = np.sqrt(np.sum(error[:, 0:3] ** 2) / len(time))
    RMS_vel = np.sqrt(np.sum(error[:, 3:6] ** 2) / len(time))

    return RMS_pos, RMS_vel
# for i in range(n):
#     Qeci = Q[i] * np.diag([1., 1., 1.])
#     Qric = 0 * np.diag([1., 1., 1.])
#     alpha = 1e-2  # to tune (in range [1e-4, 1])
#     filter_params = {
#         'Qeci': Qeci,
#         'Qric': Qric,
#         'alpha': alpha,
#         'gap_seconds': gap_seconds
#     }
#     # Radar A
#     rms_rad_pos_a[i], rms_rad_vel_a[i] = analysis2(
#         state_truth_a, state_rad_a, meas_rad_a, sensor_params_rad_a,
#         int_params, filter_params, t_rad_a)
#     rms_opt_pos_a[i], rms_opt_vel_a[i] = analysis2(
#         state_truth_a, state_opt_a, meas_opt_a, sensor_params_opt_a,
#         int_params, filter_params, t_opt_a)
#     rms_rad_pos_b[i], rms_rad_vel_b[i] = analysis2(
#         state_truth_b, state_rad_b, meas_rad_b, sensor_params_rad_b,
#         int_params, filter_params, t_rad_b)
#     rms_opt_pos_b[i], rms_opt_vel_b[i] = analysis2(
#         state_truth_b, state_opt_b, meas_opt_b, sensor_params_opt_b,
#         int_params, filter_params, t_opt_b)
#
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.loglog(Q, rms_rad_pos_a)
# ax1.loglog(Q, rms_opt_pos_a)
# ax1.loglog(Q, rms_rad_pos_b)
# ax1.loglog(Q, rms_opt_pos_b)
# ax1.set_xlabel('Qeci component [m^2/s^4]')
# ax1.set_ylabel('3D position RMS [m]')
# ax1.legend(['Radar A', 'Optical A', 'Radar B', 'Optical B'])
#
# ax2.loglog(Q, rms_rad_vel_a)
# ax2.loglog(Q, rms_opt_vel_a)
# ax2.loglog(Q, rms_rad_vel_b)
# ax2.loglog(Q, rms_opt_vel_b)
# ax2.set_xlabel('Qeci component [m^2/s^4]')
# ax2.set_ylabel('3D velocity RMS [m/s]')
# ax2.legend(['Radar A', 'Optical A', 'Radar B', 'Optical B'])
#
# plt.tight_layout()
# plt.savefig('plots/state_noise_compensation.png')
