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
    arcsec2rad = (1. / 3600.) * np.pi / 180.
    filter_output = util.ukf(state, meas, sensor,
                             integ, filters, None)
    # extract covars and state
    covars = np.vstack(
        [(filter_output[t])['covar'].reshape(1, 36) for t in time])
    state = np.vstack(
        [(filter_output[t])['state'].reshape(1, 6) for t in time])
    # dummy = len((filter_output[time[0]])['resids'])
    # resids = np.vstack(
    #     [(filter_output[t])['resids'].reshape(1, dummy) for t in time])
    state_ric_pos = np.zeros((len(time), 3))
    # state_ric_vel = np.zeros((len(time), 3))

    truth_state_ric_pos = np.zeros((len(time), 3))
    # truth_state_ric_vel = np.zeros((len(time), 3))

    cov_pos_ric = np.zeros((len(time), 9))
    # cov_vel_ric = np.zeros((len(time), 9))

    for i in range(len(time)):
        # int state
        state_ric_pos[i, :] = util.eci2ric(
            state[i, 0:3], state[i, 3:6], state[i, :3]).reshape(1, 3)
        # state_ric_vel[i, :] = util.eci2ric(
        #     state[i, 0:3], state[i, 3:6], state[i, 3:6]).reshape(1, 3)
        # true state
        truth_state_ric_pos[i, :] = util.eci2ric(
            truth_state[i, 0:3], truth_state[i, 3:6],
            truth_state[i, :3]).reshape(1, 3)
        # truth_state_ric_vel[i, :] = util.eci2ric(
        #     truth_state[i, 0:3], truth_state[i, 3:6],
        #     truth_state[i, 3:6]).reshape(1, 3)
        # covariance
        cov = covars[i, :].reshape(6, 6)
        cov_pos_ric[i, :] = (util.eci2ric(
            state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)
        # cov_vel_ric[i, :] = (util.eci2ric(
        #     state[:, 0:3], state[:, 3:6], cov[3:6, 3:6])).reshape(1, 9)

    # sigma bound
    sigma_bound_pos = 3 * np.sqrt(cov_pos_ric)
    # sigma_bound_vel = 3 * np.sqrt(np.sum(np.diag(cov_vel_ric)))
    # residuals
    resid = truth_state - state
    # RMS_pos_ = np.sqrt(np.sum(resids[:, 0:3] ** 2) / len(time))
    # RMS_vel_ = np.sqrt(np.sum(resids[:, 3:6] ** 2) / len(time))
    # print(RMS_pos_)
    # print(RMS_vel_)
    RMS_pos = np.sqrt(np.sum(resid[:, 0:3]**2)/len(time))
    RMS_vel = np.sqrt(np.sum(resid[:, 3:6]**2)/len(time))

    if plotting:
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        plt.rc('legend', fontsize=14)  # legend fontsize
        # plotting 1
        time_rel = (time - time[0]) / 3600
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(time_rel, resid[:, 0])
        ax1.plot(time_rel, sigma_bound_pos[:, 0], 'k:')
        ax1.plot(time_rel, -sigma_bound_pos[:, 0], 'k:')
        ax1.grid()
        ax2.plot(time_rel, resid[:, 1])
        ax2.plot(time_rel, sigma_bound_pos[:, 4], 'k:')
        ax2.plot(time_rel, -sigma_bound_pos[:, 4], 'k:')
        ax2.grid()
        ax3.plot(time_rel, resid[:, 2])
        ax3.plot(time_rel, sigma_bound_pos[:, 8], 'k:')
        ax3.plot(time_rel, -sigma_bound_pos[:, 8], 'k:')
        ax3.grid()
        plt.tight_layout()
        plt.savefig('plots/' + name + '_error.png')

        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
              (ax9, ax10, ax11, ax12)) = plt.subplots(
            3, 4, figsize=(12, 8))
        # R direction
        ax1.plot(time_rel[:31], resid[:31, 0])
        ax1.plot(time_rel[:31], sigma_bound_pos[:31, 0], 'k:')
        ax1.plot(time_rel[:31], -sigma_bound_pos[:31, 0], 'k:')
        ax1.grid()
        ax1.set_ylabel('R-direction error [m]')
        ax2.plot(time_rel[31:62], resid[31:62, 0])
        ax2.plot(time_rel[31:62], sigma_bound_pos[31:62, 0], 'k:')
        ax2.plot(time_rel[31:62], -sigma_bound_pos[31:62, 0], 'k:')
        ax2.grid()
        ax3.plot(time_rel[62:93], resid[62:93, 0])
        ax3.plot(time_rel[62:93], sigma_bound_pos[62:93, 0], 'k:')
        ax3.plot(time_rel[62:93], -sigma_bound_pos[62:93, 0], 'k:')
        ax3.grid()
        ax4.plot(time_rel[93:], resid[93:, 0])
        ax4.plot(time_rel[93:], sigma_bound_pos[93:, 0], 'k:')
        ax4.plot(time_rel[93:], -sigma_bound_pos[93:, 0], 'k:')
        ax4.grid()
        # I direction
        ax5.plot(time_rel[:31], resid[:31, 1])
        ax5.set_ylabel('I-direction error [m]')
        ax5.plot(time_rel[:31], sigma_bound_pos[:31, 4], 'k:')
        ax5.plot(time_rel[:31], -sigma_bound_pos[:31, 4], 'k:')
        ax5.grid()
        ax6.plot(time_rel[31:62], resid[31:62, 1])
        ax6.plot(time_rel[31:62], sigma_bound_pos[31:62, 4], 'k:')
        ax6.plot(time_rel[31:62], -sigma_bound_pos[31:62, 4], 'k:')
        ax6.grid()
        ax7.plot(time_rel[62:93], resid[62:93, 1])
        ax7.plot(time_rel[62:93], sigma_bound_pos[62:93, 4], 'k:')
        ax7.plot(time_rel[62:93], -sigma_bound_pos[62:93, 4], 'k:')
        ax7.grid()
        ax8.plot(time_rel[93:], resid[93:, 1])
        ax8.plot(time_rel[93:], sigma_bound_pos[93:, 4], 'k:')
        ax8.plot(time_rel[93:], -sigma_bound_pos[93:, 4], 'k:')
        ax8.grid()
        # C direction
        ax9.plot(time_rel[:31], resid[:31, 2])
        ax9.plot(time_rel[:31], sigma_bound_pos[:31, 8], 'k:')
        ax9.plot(time_rel[:31], -sigma_bound_pos[:31, 8], 'k:')
        ax9.set_ylabel('C-direction error [m]')
        ax9.set_xlabel('Time [hours]')
        ax9.grid()
        ax10.plot(time_rel[31:62], resid[31:62, 2])
        ax10.plot(time_rel[31:62], sigma_bound_pos[31:62, 8], 'k:')
        ax10.plot(time_rel[31:62], -sigma_bound_pos[31:62, 8], 'k:')
        ax10.set_xlabel('Time [hours]')
        ax10.grid()
        ax11.plot(time_rel[62:93], resid[62:93, 2])
        ax11.plot(time_rel[62:93], sigma_bound_pos[62:93, 8], 'k:')
        ax11.plot(time_rel[62:93], -sigma_bound_pos[62:93, 8], 'k:')
        ax11.set_xlabel('Time [hours]')
        ax11.grid()
        ax12.plot(time_rel[93:], resid[93:, 2])
        ax12.plot(time_rel[93:], sigma_bound_pos[93:, 8], 'k:')
        ax12.plot(time_rel[93:], -sigma_bound_pos[93:, 8], 'k:')
        ax12.set_xlabel('Time [hours]')
        ax12.grid()
        plt.tight_layout()
        plt.savefig('plots/' + name + '_error_sparse.png')

        if mtype == 'rad':
            meas_val = np.asarray(meas['Yk_list']).reshape(124, 3)
            resid = (state_ric_pos - meas_val)
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
                (ax9, ax10, ax11, ax12)) = plt.subplots(
                3, 4, figsize=(12, 8))
            # R direction
            ax1.scatter(time_rel[:31], resid[:31, 0])
            ax1.grid()
            ax1.set_ylabel('R-direction residual [m]')
            ax2.scatter(time_rel[31:62], resid[31:62, 0])
            ax2.grid()
            ax3.scatter(time_rel[62:93], resid[62:93, 0])
            ax3.grid()
            ax4.scatter(time_rel[93:], resid[93:, 0])
            ax4.grid()
            # I direction
            ax5.scatter(time_rel[:31], resid[:31, 1])
            ax5.set_ylabel('I-direction residual [m]')
            ax5.grid()
            ax6.scatter(time_rel[31:62], resid[31:62, 1])
            ax6.grid()
            ax7.scatter(time_rel[62:93], resid[62:93, 1])
            ax7.grid()
            ax8.scatter(time_rel[93:], resid[93:, 1])
            ax8.grid()
            # C direction
            ax9.scatter(time_rel[:31], resid[:31, 2])
            ax9.set_ylabel('C-direction residual [m]')
            ax9.set_xlabel('Time [hours]')
            ax9.grid()
            ax10.scatter(time_rel[31:62], resid[31:62, 2])
            ax10.set_xlabel('Time [hours]')
            ax10.grid()
            ax11.scatter(time_rel[62:93], resid[62:93, 2])
            ax11.set_xlabel('Time [hours]')
            ax11.grid()
            ax12.scatter(time_rel[93:], resid[93:, 2])
            ax12.set_xlabel('Time [hours]')
            ax12.grid()
            plt.tight_layout()
            plt.savefig('plots/' + name + '_residuals.png')

        elif mtype == 'opt':
            ra, dec = xyz_to_radec_rad(state_ric_pos, time_rel*3600)
            meas_val = np.asarray(meas['Yk_list'])
            resid_ra = np.abs(ra - meas_val[:, 0]) * arcsec2rad
            resid_dec = np.abs(dec - meas_val[:, 1]) * arcsec2rad
            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(
                2, 4, figsize=(16, 10))
            # R direction
            ax1.scatter(time_rel[:31], resid_ra[:31, 0])
            ax1.set_ylabel('RA residuals (arcsec)')
            ax1.grid()
            ax2.scatter(time_rel[31:62], resid_ra[31:62, 0])
            ax2.grid()
            ax3.scatter(time_rel[62:93], resid_ra[62:93, 0])
            ax3.grid()
            ax4.scatter(time_rel[93:], resid_ra[93:, 0])
            ax4.grid()
            # I direction
            ax5.scatter(time_rel[:31], resid_dec[:31, 1])
            ax5.set_ylabel('DEC residuals (arcsec)')
            ax5.set_xlabel('Time [hours]')
            ax5.grid()
            ax6.scatter(time_rel[31:62], resid_dec[31:62, 1])
            ax6.set_xlabel('Time [hours]')
            ax6.grid()
            ax7.scatter(time_rel[62:93], resid_dec[62:93, 1])
            ax7.set_xlabel('Time [hours]')
            ax7.grid()
            ax8.scatter(time_rel[93:], resid_dec[93:, 1])
            ax8.set_xlabel('Time [hours]')
            ax8.grid()
            plt.tight_layout()
            plt.savefig('plots/' + name + '_residuals.png')

    return RMS_pos, RMS_vel


def xyz_to_radec_rad(pos, time):
    # Calculate the Declination in radians
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    dec = np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    # Calculate the Right Ascension in radians
    theta0 = 0.
    dtheta = 7.2921158553e-5
    # RA is the arctan of Y divided by X
    ra = np.arctan2(y, x) - np.arange(len(time))*10*dtheta

    # Ensure RA is between 0 and 2π
    for i in range(len(x)):
        if ra[i] < 0:
            ra[i] += 2 * math.pi

    return ra, dec


# Radar A
# Set process noise
Qeci = 0 * np.diag([1., 1., 1.])
Qric = 0 * np.diag([1., 1., 1.])
alpha = 1e-1                            # to tune (in range [1e-4, 1])
filter_params = {
    'Qeci': Qeci,
    'Qric': Qric,
    'alpha': alpha,
    'gap_seconds': gap_seconds
}
rms_rad_pos_a, rms_rad_vel_a = analysis(
    state_truth_a, state_rad_a, meas_rad_a, sensor_params_rad_a,
    int_params, filter_params, t_rad_a, 'radar_a', 'rad', 1)
print(rms_rad_pos_a)
print(rms_rad_vel_a)
# Optical A
# Set process noise
Qeci = 0 * np.diag([1., 1., 1.])
Qric = 0 * np.diag([1., 1., 1.])
alpha = 1e-1                            # to tune (in range [1e-4, 1])
filter_params = {
    'Qeci': Qeci,
    'Qric': Qric,
    'alpha': alpha,
    'gap_seconds': gap_seconds
}
rms_opt_pos_a, rms_opt_vel_a = analysis(
    state_truth_a, state_opt_a, meas_opt_a, sensor_params_opt_a,
    int_params, filter_params, t_opt_a, 'optical_a', 'opt', 1)
print(rms_opt_pos_a)
print(rms_opt_vel_a)

# Radar B
# Set process noise
Qeci = 0 * np.diag([1., 1., 1.])
Qric = 1e-12 * np.diag([1., 1., 1.])
alpha = 1e-1                            # to tune (in range [1e-4, 1])
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
Qeci = 0 * np.diag([1., 1., 1.])
Qric = 1e-15 * np.diag([1., 1., 1.])
alpha = 1e-1                            # to tune (in range [1e-4, 1])
filter_params = {
    'Qeci': Qeci,
    'Qric': Qric,
    'alpha': alpha,
    'gap_seconds': gap_seconds
}
rms_opt_pos_b, rms_opt_vel_b = analysis(
    state_truth_b, state_opt_b, meas_opt_b, sensor_params_opt_b,
    int_params, filter_params, t_opt_b, 'optical_b', 'opt', 1)
print(rms_opt_pos_b)
print(rms_opt_vel_b)

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
#     rms_rad_pos_a[i], rms_rad_vel_a[i] = analysis(
#         state_truth_a, state_rad_a, meas_rad_a, sensor_params_rad_a,
#         int_params, filter_params, t_rad_a, 'radar_a', 'rad', 0)
#     rms_opt_pos_a[i], rms_opt_vel_a[i] = analysis(
#         state_truth_a, state_opt_a, meas_opt_a, sensor_params_opt_a,
#         int_params, filter_params, t_opt_a, 'optical_a', 'opt', 0)
#     rms_rad_pos_b[i], rms_rad_vel_b[i] = analysis(
#         state_truth_b, state_rad_b, meas_rad_b, sensor_params_rad_b,
#         int_params, filter_params, t_rad_b, 'radar_b', 'rad', 0)
#     rms_opt_pos_b[i], rms_opt_vel_b[i] = analysis(
#         state_truth_b, state_opt_b, meas_opt_b, sensor_params_opt_b,
#         int_params, filter_params, t_opt_b, 'optical_b', 'opt', 0)
#
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     ax1.loglog(Q, rms_rad_pos_a)
#     ax1.loglog(Q, rms_opt_pos_a)
#     ax1.loglog(Q, rms_rad_pos_b)
#     ax1.loglog(Q, rms_opt_pos_b)
#     ax1.set_xlabel('Qeci component [m^2/s^4]')
#     ax1.set_ylabel('3D position RMS [m]')
#     ax1.legend(['Radar A', 'Optical A', 'Radar B', 'Optical B'])
#
#     ax2.loglog(Q, rms_rad_vel_a)
#     ax2.loglog(Q, rms_opt_vel_a)
#     ax2.loglog(Q, rms_rad_vel_b)
#     ax2.loglog(Q, rms_opt_vel_b)
#     ax2.set_xlabel('Qeci component [m^2/s^4]')
#     ax2.set_ylabel('3D velocity RMS [m/s]')
#     ax2.legend(['Radar A', 'Optical A', 'Radar B', 'Optical B'])
#
#     plt.tight_layout()
#     plt.savefig('plots/state_noise_compensation.png')
