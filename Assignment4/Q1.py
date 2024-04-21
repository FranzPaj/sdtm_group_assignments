import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from pycode.TudatPropagator import propagate_orbit, tudat_initialize_bodies
from pycode.EstimationUtilities import compute_measurement, ukf, eci2ric
from tudatpy.astro.time_conversion import epoch_from_date_time_components


def diff_meas(t_meas, y_meas, t0, x0, state_params, int_params, sensor_params):
    t_sim, x_sim = propagate_orbit(x0, [t0, t_meas[-1]], state_params, int_params)
    res = np.zeros((len(t_meas), 2))
    for i, time in enumerate(t_meas):
        sim_index = np.argmin(np.abs(t_sim - time))
        x_now = x_sim[sim_index]
        sim_meas = compute_measurement(t_sim[sim_index], x_now, sensor_params)
        res[i] = (y_meas[i] - sim_meas).flatten()
        # if t_sim[sim_index] != time:
        #     print(f"WARNING: measurement time {time} does not appear in the simulated times.")
        #     print(f"Closest time found: {t_sim[sim_index]}")
    return res


def prefit_residuals(state_params, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds):
    int_params = {'tudat_integrator': 'rk4', 'step': 10.}
    t_meas = measurement['tk_list']
    y_meas = measurement['Yk_list']
    utc = default_state['UTC']
    t0 = epoch_from_date_time_components(
        utc.year, utc.month, utc.day, utc.hour, utc.minute, float(utc.second))
    residuals_list = diff_meas(t_meas, y_meas, t0, state_params['state'], state_params, int_params, sensor_params)
    res = {}
    for i, residual in enumerate(residuals_list):
        res[t_meas[i]] = residual
    return res


def postfit_residuals(state_params, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds):
    int_params = {'tudat_integrator': 'rk4', 'step': 10.}
    filter_params = {'Qeci': q_eci, 'Qric': q_ric, 'alpha': alpha, 'gap_seconds': gap_seconds}
    bodies = tudat_initialize_bodies(["Earth", "Sun", "Moon"])
    ukf_res = ukf(state_params, measurement, sensor_params, int_params, filter_params, bodies)
    res = {}
    for time, entry in ukf_res.items():
        res[time] = entry['resids'].flatten()
    return res


def resids(fit_type, state_params, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds):
    if fit_type == 'pre':
        return prefit_residuals(state_params, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds)
    else:
        return postfit_residuals(state_params, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds)


def score(residuals):
    res_list = np.vstack(list(residuals.values()))
    res_ra = np.sqrt(np.mean(res_list[:, 0] ** 2))
    res_dec = np.sqrt(np.mean(res_list[:, 1] ** 2))
    res = res_dec + res_ra
    return res


def f_to_optimize(fit_type, changed_type, state_params, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds):
    def score_function(new_value):
        new_state = state_params.copy()
        new_state[changed_type] = new_value
        res_dict = resids(fit_type, new_state, measurement, sensor_params, q_eci, q_ric, alpha, gap_seconds)
        res = score(res_dict)
        print(f'{changed_type}: {new_value}')
        print(f'score: {res}')
        return res

    return score_function


if __name__ == "__main__":
    with open('data/states_updated/q1_meas_objchar_91762.pkl', 'rb') as file:
        data = pkl.load(file)
    default_state = data[0]
    sensor_params = data[1]
    updated_data = data[2]
    meas_times = updated_data['tk_list']

    fit_type = 'pre'

    q_eci = 0 * np.eye(3)
    q_ric = 5e-14 * np.eye(3)
    alpha = 1e-1
    gap_seconds = 24 * 3600

    new_state = default_state.copy()

    if fit_type == 'pre':
        new_state['area'] = np.sqrt(130.13 / 1.3)
        new_state['mass'] = 100 / np.sqrt(130.13 / 1.3)
    else:
        new_state['area'] = 10.065
        new_state['mass'] = 9.9352

    # Compute and plot the initial residuals
    res_dict = resids(
        fit_type, default_state, updated_data, sensor_params, q_eci, q_ric, alpha, gap_seconds
    )
    residuals = np.vstack(list(res_dict.values()))

    time_hr = (meas_times - meas_times[0]) / 3600
    fig = plt.figure()
    plt.plot(time_hr, np.rad2deg(residuals[:, 0]), '.', label="Right ascension")
    plt.plot(time_hr, np.rad2deg(residuals[:, 1]), '.', label="Declination")
    plt.xlabel("Time [hr]")
    plt.ylabel("Angular difference [deg]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Compute and plot the final (optimized) residuals
    opt_res_dict = resids(
        fit_type, new_state, updated_data, sensor_params, q_eci, q_ric, alpha, gap_seconds
    )
    opt_res = np.vstack(list(opt_res_dict.values()))

    fig = plt.figure()
    plt.plot(time_hr, np.rad2deg(opt_res[:, 0]), '.', label="Right ascension")
    plt.plot(time_hr, np.rad2deg(opt_res[:, 1]), '.', label="Declination")
    plt.xlabel("Time [hr]")
    plt.ylabel("Angular difference [deg]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Compare the prefit and the postfit residuals
    for fit_type in ['pre', 'post']:
        res_dict = resids(
            fit_type, default_state, updated_data, sensor_params, q_eci, q_ric, alpha, gap_seconds
        )
        print(f"{fit_type} residual: {score(res_dict)}")

    # Compare the results using the three parameters
    lis_par = ['mass', 'area', 'Cr']
    dic_res = {}
    for par_type in lis_par:
        f_opt = f_to_optimize(
            fit_type, par_type, default_state, updated_data, sensor_params, q_eci, q_ric, alpha, gap_seconds)
        x0 = default_state[par_type]
        popt = minimize_scalar(
            f_opt, method="bounded", options={'maxiter': 100, 'disp': True}, bounds=(1e-3 * x0, 1e3 * x0))
        dic_res[par_type] = popt

    # To get more than just the residuals, it is necessary to rerun the UKF and get the whole output
    int_params = {'tudat_integrator': 'rk4', 'step': 10.}
    filter_params = {'Qeci': q_eci, 'Qric': q_ric, 'alpha': alpha, 'gap_seconds': gap_seconds}
    bodies = tudat_initialize_bodies(["Earth", "Sun", "Moon"])
    if fit_type == 'pre':  # If the fit type was on pre, change to have the correct optimal parameters
        new_state['area'] = 10.065
        new_state['mass'] = 9.9352
    ukf_final = ukf(new_state, updated_data, sensor_params, int_params, filter_params, bodies)
    times = np.array(list(ukf_final.keys()))
    covars = np.vstack(
        [(ukf_final[t])['covar'].reshape(1, 36) for t in times])
    state = np.vstack(
        [(ukf_final[t])['state'].reshape(1, 6) for t in times])
    cov_pos_ric = np.zeros((len(times), 9))
    for i in range(len(times)):
        cov = covars[i, :].reshape(6, 6)
        cov_pos_ric[i, :] = (eci2ric(state[i, 0:3], state[i, 3:6], cov[:3, :3])).reshape(1, 9)

    # sigma bound
    sigma_bound_pos = 3 * np.sqrt(cov_pos_ric)

    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=14)  # legend fontsize
    # plotting 1
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
          (ax9, ax10, ax11, ax12)) = plt.subplots(
        3, 4, figsize=(12, 8))
    # R direction
    ax1.plot(time_hr[:60], sigma_bound_pos[:60, 0], 'k:')
    ax1.plot(time_hr[:60], -sigma_bound_pos[:60, 0], 'k:')
    ax1.grid()
    ax1.set_ylabel('R-direction [m]', size=18)
    ax2.plot(time_hr[60:300], sigma_bound_pos[60:300, 0], 'k:')
    ax2.plot(time_hr[60:300], -sigma_bound_pos[60:300, 0], 'k:')
    ax2.grid()
    ax3.plot(time_hr[300:360], sigma_bound_pos[300:360, 0], 'k:')
    ax3.plot(time_hr[300:360], -sigma_bound_pos[300:360, 0], 'k:')
    ax3.grid()
    ax4.plot(time_hr[360:], sigma_bound_pos[360:, 0], 'k:')
    ax4.plot(time_hr[360:], -sigma_bound_pos[360:, 0], 'k:')
    ax4.grid()
    # I direction
    ax5.set_ylabel('I-direction [m]', size=18)
    ax5.plot(time_hr[:60], sigma_bound_pos[:60, 4], 'k:')
    ax5.plot(time_hr[:60], -sigma_bound_pos[:60, 4], 'k:')
    ax5.grid()
    ax6.plot(time_hr[60:300], sigma_bound_pos[60:300, 4], 'k:')
    ax6.plot(time_hr[60:300], -sigma_bound_pos[60:300, 4], 'k:')
    ax6.grid()
    ax7.plot(time_hr[300:360], sigma_bound_pos[300:360, 4], 'k:')
    ax7.plot(time_hr[300:360], -sigma_bound_pos[300:360, 4], 'k:')
    ax7.grid()
    ax8.plot(time_hr[360:], sigma_bound_pos[360:, 4], 'k:')
    ax8.plot(time_hr[360:], -sigma_bound_pos[360:, 4], 'k:')
    ax8.grid()
    # C direction
    ax9.plot(time_hr[:60], sigma_bound_pos[:60, 8], 'k:')
    ax9.plot(time_hr[:60], -sigma_bound_pos[:60, 8], 'k:')
    ax9.set_ylabel('C-direction [m]', size=18)
    ax9.set_xlabel('Time [hr]', size=18)
    ax9.grid()
    ax10.plot(time_hr[60:300], sigma_bound_pos[60:300, 8], 'k:')
    ax10.plot(time_hr[60:300], -sigma_bound_pos[60:300, 8], 'k:')
    ax10.set_xlabel('Time [hr]', size=18)
    ax10.grid()
    ax11.plot(time_hr[300:360], sigma_bound_pos[300:360, 8], 'k:')
    ax11.plot(time_hr[300:360], -sigma_bound_pos[300:360, 8], 'k:')
    ax11.set_xlabel('Time [hr]', size=18)
    ax11.grid()
    ax12.plot(time_hr[360:], sigma_bound_pos[360:, 8], 'k:')
    ax12.plot(time_hr[360:], -sigma_bound_pos[360:, 8], 'k:')
    ax12.set_xlabel('Time [hr]', size=18)
    ax12.grid()
    plt.tight_layout()
