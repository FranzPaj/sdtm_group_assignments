import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from datetime import datetime

from pycode.ConjunctionUtilities import *
from pycode.EstimationUtilities import *
from pycode.TudatPropagator import *
from tudatpy.astro import two_body_dynamics
from tudatpy.astro import element_conversion
from tudatpy.astro import time_conversion
from tudatpy.numerical_simulation import environment_setup


mu = spice.get_body_gravitational_parameter('Earth')

def angles_only_gauss(L, t1, t2, t3, r_site):
    tau1 = t1 - t2
    tau3 = t3 - t2

    a1 = tau3 / (tau3 - tau1)
    a3 = -tau1 / (tau3 - tau1)
    a1u = tau3 * ((tau3 - tau1) ** 2 - tau3 ** 2) / (6 * (tau3 - tau1))
    a3u = - tau1 * ((tau3 - tau1) ** 2 - tau1 ** 2) / (6 * (tau3 - tau1))

    L = L.transpose()
    r_site = r_site.transpose()
    L_inv = np.linalg.inv(L)
    M = np.matmul(L_inv, r_site)

    d1 = M[1, 0] * a1 - M[1, 1] + M[1, 2] * a3
    d2 = M[1, 0] * a1u + M[1, 2] * a3u

    C = np.dot(L[:, 1], r_site[:, 1])

    coefficient = [1, 0, -(d1 ** 2 + 2 * C * d1 + np.linalg.norm(r_site[:, 1]) ** 2),
                   0, 0, -2 * mu * (C * d2 + d1 * d2), 0, 0, -mu ** 2 * d2 ** 2]

    roots = np.roots(coefficient)

    real_positive_roots = [root.real for root in roots if np.isclose(root.imag, 0, atol=1e-10)] # and root.real > 0]

    range_solution = np.zeros((3, len(real_positive_roots)))
    r_eci = np.zeros((3, 3*len(real_positive_roots)))
    iter = 0
    for root in real_positive_roots:
        u = mu / root ** 3
        c1 = a1 + a1u * u
        c2 = -1
        c3 = a3 + a3u * u
        c_vec = np.array([c1, c2, c3]).reshape(3, 1)
        c_range = np.matmul(M, -1*c_vec).reshape(3,)
        range_solution[:, iter] = c_range/c_vec.reshape(3,)
        for i in range(3):
            r_eci[:, 3*iter+i] = range_solution[i, iter]*L[:, i] + r_site[:, i]
        iter += 1

    return r_eci, range_solution


def gooding_angles_only(L, t1, t2, t3, r_site, range1, range3):
    L = L.transpose()
    r_site = r_site.transpose()

    r_1 = range1*L[:, 0] + r_site[:, 0]
    r_3 = range3*L[:, 2] + r_site[:, 2]

    lambert_arc_ephemeris_initial = lambert_arc(r_1, r_3, t1, t3)
    r_2 = lambert_arc_ephemeris_initial.cartesian_state(t2)[0:3]
    range2 = np.linalg.norm(r_2 - r_site[:, 1])
    L2_lambert = (r_2 - r_site[:, 1].reshape(3))/range2

    distance, x_hat, y_hat, origin_ref = point_to_line_distance(L2_lambert * range2, L[:, 1].reshape(3,))

    it = 0
    delta = 1e-5
    total_range1 = np.array([])
    total_range3 = np.array([])
    while distance > 10 and it < 200000:
        # partial derivatives as finite difference
        range1_new = range1 + delta
        range3_new = range3 + delta
        r1_new = range1_new*L[:, 0] + r_site[:, 0]
        r3_new = range3_new*L[:, 2] + r_site[:, 2]
        lambert_arc_new1 = lambert_arc(r1_new, r_3, t1, t3)
        lambert_arc_new3 = lambert_arc(r_1, r3_new, t1, t3)
        r2_new1 = lambert_arc_new1.cartesian_state(t2)[0:3]
        r2_new3 = lambert_arc_new3.cartesian_state(t2)[0:3]
        range2_new1 = np.linalg.norm(r2_new1 - r_site[:, 1].reshape(3, ))
        range2_new3 = np.linalg.norm(r2_new3 - r_site[:, 1].reshape(3, ))
        L2_new1 = (r2_new1 - r_site[:, 1].reshape(3, )) / range2_new1
        L2_new3 = (r2_new3 - r_site[:, 1].reshape(3)) / range2_new3

        dist_x1, dist_y1 = distance_from_new_origin(x_hat, y_hat, origin_ref, L2_new1 * range2_new1)
        dist_x2, dist_y2 = distance_from_new_origin(x_hat, y_hat, origin_ref, L2_new3 * range2_new3)

        fx = (dist_x1 - distance)/delta
        fy = (dist_x2 - distance)/delta
        gx = dist_y1/delta
        gy = dist_y2/delta

        D = fx*gy - fy*gx
        delta_rho1 = -1/D * distance * gy
        delta_rho2 = 1/D * distance * gx

        range1 = range1 + delta_rho1
        range3 = range3 + delta_rho2
        r_1 = range1 * L[:, 0] + r_site[:, 0]
        r_3 = range3 * L[:, 2] + r_site[:, 2]
        lambert_arc_ephemeris = lambert_arc(r_1, r_3, t1, t3)
        r_2 = lambert_arc_ephemeris.cartesian_state(t2)[0:3]
        range2 = np.linalg.norm(r_2 - r_site[:, 1].reshape(3, ))
        L2_lambert = (r_2 - r_site[:, 1].reshape(3, )) / range2
        distance, x_hat, y_hat, origin_ref = point_to_line_distance(L2_lambert * range2, L[:, 1].reshape(3, ))
        if distance < 120:
            total_range1 = np.append(total_range1, range1)
            total_range3 = np.append(total_range3, range3)
        print(distance)
        it += 1

    return (range1, range2, range3, it, lambert_arc_ephemeris, total_range1, total_range3,
            lambert_arc_ephemeris_initial)


def lambert_arc(initial_pos, final_pos, t1, t2):
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_pos, final_pos, t2 - t1, mu)

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = np.concatenate((initial_pos, np.array([0, 0, 0])))
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(lambert_arc_initial_state, mu)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, t1,
                                              mu), "")

    return kepler_ephemeris


def point_to_line_distance(point, line_vec):
    angle = np.arccos(np.dot(point, line_vec)/(np.linalg.norm(point)*np.linalg.norm(line_vec)))
    distance = np.linalg.norm(point)*np.sin(angle)
    origin_reference = np.dot(line_vec, point)

    # define plane
    x = point - line_vec*(np.dot(point, line_vec))
    x_versor = x/np.linalg.norm(x)
    y_versor = np.cross(x_versor, line_vec)

    return distance, x_versor, y_versor, origin_reference


def distance_from_new_origin(x_hat, y_hat, origin_reference, vector):
    x_dist = np.dot(x_hat, vector)
    y_dist = np.dot(y_hat, vector)
    return x_dist, y_dist

def angles_double_r(L, t1, t2, t3, r_site, r1_est, r2_est):
    L = L.transpose()
    r_site = r_site.transpose()

    tau1 = t1 - t2
    tau3 = t3 - t2

    c1_in = 2*np.dot(L[:, 0], r_site[:, 0])
    c2_in = 2*np.dot(L[:, 1], r_site[:, 1])

    iter = 0
    delta_r1 = 100000
    delta_r2 = 100000

    def evaluate_F(dr1, dr2):
        r1_est_d = r1_est + dr1
        r2_est_d = r2_est + dr2
        rho1 = (-c1_in + np.sqrt(c1_in ** 2 - 4 * (np.linalg.norm(r_site[:, 0]) ** 2 - r1_est_d ** 2))) / 2
        rho2 = (-c2_in - np.sqrt(c2_in ** 2 - 4 * (np.linalg.norm(r_site[:, 1]) ** 2 - r2_est_d ** 2))) / 2

        r1 = rho1 * L[:, 0] + r_site[:, 0]
        r2 = rho2 * L[:, 1] + r_site[:, 1]

        W = np.cross(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))

        rho3 = (-np.dot(r_site[:, 2], W)) / (np.dot(L[:, 2], W))

        r3 = rho3 * L[:, 2] + r_site[:, 2]

        # for cycle j = 2,3, k = 1,2
        # J = 2, k = 1
        cos_dv_21 = np.dot(r2, r1) / (np.linalg.norm(r2) * np.linalg.norm(r1))
        sin_dv_21 = np.sqrt(1 - cos_dv_21**2)
        # j = 3, k = 2
        cos_dv_32 = np.dot(r3, r2) / (np.linalg.norm(r3) * np.linalg.norm(r2))
        sin_dv_32 = np.sqrt(1 - cos_dv_32**2)
        # j = 3, k = 1
        cos_dv_31 = np.dot(r3, r1) / (np.linalg.norm(r3) * np.linalg.norm(r1))
        sin_dv_31 = np.sqrt(1 - cos_dv_31**2)

        # assuming prograde orbit (dv < 180°)
        c1 = np.linalg.norm(r1) * sin_dv_31 / (np.linalg.norm(r2) * sin_dv_32)
        c3 = np.linalg.norm(r1) * sin_dv_21 / (np.linalg.norm(r3) * sin_dv_32)
        p = (c3 * np.linalg.norm(r3) - c1 * np.linalg.norm(r2) + np.linalg.norm(r1)) / (-c1 + c3 + 1)

        e_cos_v1 = p / np.linalg.norm(r1) - 1
        e_cos_v2 = p / np.linalg.norm(r2) - 1
        e_cos_v3 = p / np.linalg.norm(r3) - 1

        if np.arccos(cos_dv_21) != np.pi:
            e_sin_v2 = (-cos_dv_21 * e_cos_v2 + e_cos_v1) / sin_dv_21
        else:
            e_sin_v2 = (cos_dv_32 * e_cos_v2 - e_cos_v3) / sin_dv_31
        print(sin_dv_21)
        print(e_cos_v2, e_sin_v2)
        e = np.sqrt(e_cos_v2 ** 2 + e_sin_v2 ** 2)
        print(e)
        a = p / (1 - e ** 2)
        n = np.sqrt(mu / a ** 3)
        S = np.linalg.norm(r2) / p * np.sqrt(1 - e ** 2) * e_sin_v2
        C = np.linalg.norm(r2) / p * (e ** 2 + e_cos_v2)

        sin_de_32 = np.linalg.norm(r3) / np.sqrt(a * p) * sin_dv_32 - np.linalg.norm(r3) / p * (1 - cos_dv_32) * S
        cos_de_32 = 1 - np.linalg.norm(r2) * np.linalg.norm(r3) / (a * p) * (1 - cos_dv_32)
        sin_de_21 = np.linalg.norm(r1) / np.sqrt(a * p) * sin_dv_21 + np.linalg.norm(r1) / p * (1 - cos_dv_21) * S
        cos_de_21 = 1 - np.linalg.norm(r2) * np.linalg.norm(r1) / (a * p) * (1 - cos_dv_21)

        de_32 = math.atan2(sin_de_32, cos_de_32)
        de_21 = math.atan2(sin_de_21, cos_de_21)

        dm_32 = de_32 + 2 * S * (np.sin(de_32 / 2)) ** 2 - C * sin_de_32
        dm_21 = -de_21 + 2 * S * np.sin(de_21 / 2) ** 2 + C * sin_de_21

        F1 = tau1 - dm_21 / n
        print(n)
        F2 = tau3 - dm_32 / n
        print(r2)
        print(np.linalg.norm(r2))
        return F1, F2, de_32, a, r1, r2, r3

    while (delta_r1 > 10000 or delta_r2 > 10000) or iter < 10000:
        F1, F2, de_32, a, r1, r2, r3 = evaluate_F(0, 0)
        Q = np.sqrt(F1**2 + F2**2)
        dr1 = 0.005*r1_est
        dr2 = 0.005*r2_est
        F1_dr1, F2_dr1, _, _, _, _, _ = evaluate_F(dr1, 0)
        F1_dr2, F2_dr2, _, _, _, _, _ = evaluate_F(0, dr2)

        dF1_dr1 = (F1_dr1 - F1)/dr1
        dF2_dr1 = (F2_dr1 - F2)/dr1

        dF1_dr2 = (F1_dr2 - F1)/dr2
        dF2_dr2 = (F2_dr2 - F2)/dr2
        delta = dF1_dr1*dF2_dr2 - dF2_dr1*dF1_dr2
        delta_1 = dF2_dr2*F1 - dF1_dr2*F2
        delta_2 = dF1_dr1*F2 - dF2_dr1*F1
        alpha_nickM = 1e-5
        delta_r1 = - delta_1/delta * alpha_nickM
        delta_r2 = - delta_2/delta * alpha_nickM
        r1_est = r1_est + delta_r1
        r2_est = r2_est + delta_r2
        iter += 1

    f = 1 - a/np.linalg.norm(r2) * (1 - np.cos(de_32))
    g = tau3 - np.sqrt(a**3/mu)*(de_32 - np.sin(de_32))
    v2 = (r3 - f*r2)/g

    return r1, r2, r3, v2


def line_of_sight(ra, dec):
    line_vec = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    return line_vec

def refine_solution(initial_state, epochs, r_site, L, sensor_params_opt, range1, range1_list, Yk):
    bodies_to_create = ['Sun', 'Earth', 'Moon']

    state_param = {'Cd': 1.2, 'Cr': 1.2, 'area': 1, 'mass': 10, 'sph_deg': 8, 'sph_ord': 8,
                   'bodies_to_create': bodies_to_create, 'central_bodies': ['Earth']}
    int_param = {'tudat_integrator': 'rk4', 'step': 50}
    tic = time.time()
    t_rso, X_rso_prop = propagate_orbit(initial_state, [epochs[0], epochs[2]], state_param, int_param)
    toc = time.time()
    print('Single propagation time [s]: ', toc - tic)
    index = np.where(t_rso == epochs[1])[0]
    r_2 = X_rso_prop[index, :3]
    index = np.where(t_rso == epochs[2])[0]
    r_3 = X_rso_prop[index, :3]

    r_2_stat = r_2 - r_site[1, :]
    r_3_stat = r_3 - r_site[2, :]
    L_2 = r_2_stat / np.linalg.norm(r_2_stat)
    L_3 = r_3_stat / np.linalg.norm(r_3_stat)

    error = np.abs(np.arccos(np.dot(L_2, L[1, :]))) + np.abs(np.arccos(np.dot(L_3, L[2, :])))

    error_min = error

    sigma_posx = 0.0001 * np.abs(initial_state[0])
    sigma_posy = 0.0001 * np.abs(initial_state[1])
    sigma_posz = 0.0001 * np.abs(initial_state[2])
    sigma_velx = 0.001 * np.abs(initial_state[3])
    sigma_vely = 0.001 * np.abs(initial_state[4])
    sigma_velz = 0.001 * np.abs(initial_state[5])

    n = 1000
    perturbed_initial_state = np.column_stack((
        np.random.normal(loc=initial_state[0], scale=sigma_posx, size=n),
        np.random.normal(loc=initial_state[1], scale=sigma_posy, size=n),
        np.random.normal(loc=initial_state[2], scale=sigma_posz, size=n),
        np.random.normal(loc=initial_state[3], scale=sigma_velx, size=n),
        np.random.normal(loc=initial_state[4], scale=sigma_vely, size=n),
        np.random.normal(loc=initial_state[5], scale=sigma_velz, size=n)
    ))

    good_states = []
    best_state = initial_state

    for i in range(n):
        new_state = perturbed_initial_state[i, :]
        t_rso, X_rso_prop = propagate_orbit(new_state, [epochs[0], epochs[2]], state_param, int_param)

        r_1 = X_rso_prop[0, :3]
        index = np.where(t_rso == epochs[1])[0]
        r_2 = X_rso_prop[index, :3]
        index = np.where(t_rso == epochs[2])[0]
        r_3 = X_rso_prop[index, :3]

        r_1_stat = r_1 - r_site[0, :]
        r_2_stat = r_2 - r_site[1, :]
        r_3_stat = r_3 - r_site[2, :]
        L_1 = r_1_stat / np.linalg.norm(r_1_stat)
        L_2 = r_2_stat / np.linalg.norm(r_2_stat)
        L_3 = r_3_stat / np.linalg.norm(r_3_stat)

        error_new = (np.abs(np.arccos(np.dot(L_1, L[0, :]))) + np.abs(np.arccos(np.dot(L_2, L[1, :]))) +
                 np.abs(np.arccos(np.dot(L_3, L[2, :]))))

        if error_new < error:
            good_states.append(new_state)
        if error_new < error_min:
            best_state = np.copy(new_state)
            error_min = np.copy(error_new)

        if i % 100 == 0:
            print('iterations perturbed propagation completed: ', i)

    # calculate std deviations in ECI
    sigma_angles = sensor_params_opt['sigma_dict']['dec']
    sigma_radial1 = np.std(range1_list)

    n = 100000
    perturbed_initial_state = np.column_stack((
        np.random.normal(loc=Yk[0][0], scale=sigma_angles, size=n),
        np.random.normal(loc=Yk[0][1], scale=sigma_angles, size=n),
        np.random.normal(loc=range1, scale=sigma_radial1, size=n)
    ))

    position_1_eci = np.zeros((n, 3))
    for i in range(n):
        line_1 = line_of_sight(perturbed_initial_state[i, 0], perturbed_initial_state[i, 1])
        vec_new = r_site[0, :] + perturbed_initial_state[i, 2] * line_1
        position_1_eci[i, :] = vec_new.reshape(3,)
        if i % 5000 == 0:
            print('Iterations std deviation estimation completed: ', i)
    std_dev_eci_x, std_dev_eci_y, std_dev_eci_z = (np.std(position_1_eci[:, 0]), np.std(position_1_eci[:, 1]),
                                                   np.std(position_1_eci[:, 2]))
    good_states_array = np.array(good_states)

    return best_state, good_states_array, np.array([std_dev_eci_x, std_dev_eci_y, std_dev_eci_z])


def plot_earth_and_vectors(vectors_site, L):
    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define Earth's radius (in m)
    earth_radius = 6371

    # Generate data for Earth's surface (a sphere)
    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = earth_radius*np.sin(theta)*np.cos(phi)
    y = earth_radius*np.sin(theta)*np.sin(phi)
    z = earth_radius*np.cos(theta)

    # Plot the surface
    ax.plot_surface(x, y, z, color='b', alpha=0.2, rstride=1, cstride=1, linewidth=0, antialiased=False)

    leg = ['LOS 1', 'LOS 2', 'LOS 3']
    col = ['r', 'g', 'b']
    for i in range(len(vectors_site[:, 0])):
        site = vectors_site[i, :]/1000
        line = L[i, :]*3000
        ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[i],
                  arrow_length_ratio=0.3, label=leg[i])

    # Set plot display parameters
    ax.set_xlim([-8000, 8000])
    ax.set_ylim([-8000, 8000])
    ax.set_zlim([-8000, 8000])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Earth Representation and Lines of Sight')
    ax.view_init(azim=-110)
    ax.legend()
    # Customize the number of ticks on each axis
    ax.xaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    ax.yaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    ax.zaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    # Show the plot
    plt.tight_layout()
    plt.savefig('./Q3_plots/Lines_of_sights.png')
    plt.show()

def plot_earth_and_ranges(vectors_site, L, range1, range3):
    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define Earth's radius (in m)
    earth_radius = 6371

    # Generate data for Earth's surface (a sphere)
    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = earth_radius*np.sin(theta)*np.cos(phi)
    y = earth_radius*np.sin(theta)*np.sin(phi)
    z = earth_radius*np.cos(theta)

    # Plot the surface
    ax.plot_surface(x, y, z, color='b', alpha=0.2, rstride=1, cstride=1, linewidth=0, antialiased=False)

    leg = ['Range obs 1', 'Range obs 3']
    col = ['r', 'b']

    site = vectors_site[0, :]/1000
    line = L[0, :]*range1/1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[0],
              arrow_length_ratio=0.1, label=leg[0])
    site = vectors_site[2, :]/1000
    line = L[2, :]*range3/1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[1],
              arrow_length_ratio=0.1, label=leg[1])

    # Set plot display parameters
    ax.set_xlim([-40000, 40000])
    ax.set_ylim([-40000, 40000])
    ax.set_zlim([-40000, 40000])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Earth Representation and Initial estimated ranges')
    ax.view_init(azim=-100)
    ax.legend()
    # Customize the number of ticks on each axis
    ax.xaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    ax.yaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    ax.zaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    # Show the plot
    plt.tight_layout()
    plt.savefig('./Q3_plots/initial_ranges.png')
    plt.show()


def plot_orbit(ephemeris_in, ephemeris_end, t1, t2, t3, r_site, L, range_in, range_end1, range_end2, range_end3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define Earth's radius (in m)
    earth_radius = 6371

    # Generate data for Earth's surface (a sphere)
    phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
    x = earth_radius * np.sin(theta) * np.cos(phi)
    y = earth_radius * np.sin(theta) * np.sin(phi)
    z = earth_radius * np.cos(theta)

    # Plot the surface
    ax.plot_surface(x, y, z, color='b', alpha=0.2, rstride=1, cstride=1, linewidth=0, antialiased=False)

    pos_in = np.zeros((10000, 3))
    pos_end = np.zeros((10000, 3))
    t = np.linspace(t1, t3, 10000)
    for i in range(10000):
        pos_in[i, :] = ephemeris_in.cartesian_state(t[i])[:3]
        pos_end[i, :] = ephemeris_end.cartesian_state(t[i])[:3]
    ax.plot(pos_in[:, 0]/1000, pos_in[:, 1]/1000, pos_in[:, 2]/1000, linestyle='dashed', color='k',
            label='First iteration')
    ax.plot(pos_end[:, 0]/1000, pos_end[:, 1]/1000, pos_end[:, 2]/1000, color='k', label='Final iteration')

    col = ['r', 'g', 'b']
    '''
    site = r_site[0, :] / 1000
    line = L[0, :] * range_in / 1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[0],
              arrow_length_ratio=0.2, linestyle='dashed', label='Range 1 initial')

    r_2 = ephemeris_in.cartesian_state(t2)[0:3]
    range2 = np.linalg.norm(r_2 - r_site[1, :])
    site = r_site[1, :] / 1000
    line = (r_2 - r_site[1, :].reshape(3))/1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[1],
              arrow_length_ratio=0.2, linestyle='dashed', label='Range 2 initial')

    site = r_site[2, :] / 1000
    line = L[2, :] * range_in / 1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[2],
              arrow_length_ratio=0.2, linestyle='dashed', label='Range 3 initial')
    '''
    site = r_site[0, :] / 1000
    line = L[0, :] * range_end1 / 1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[0],
              arrow_length_ratio=0.1, label='Range 1 final')

    site = r_site[1, :] / 1000
    line = L[1, :] * range_end2 / 1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[1],
              arrow_length_ratio=0.1, label='Range 2 final')

    site = r_site[2, :] / 1000
    line = L[2, :] * range_end3 / 1000
    ax.quiver(site[0], site[1], site[2], line[0], line[1], line[2], color=col[2],
              arrow_length_ratio=0.1, label='Range 3 final')

    # Set plot display parameters
    ax.set_xlim([-45000, 45000])
    ax.set_ylim([-45000, 45000])
    ax.set_zlim([-45000, 45000])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Earth Representation and Gooding method orbit solution')
    ax.view_init(azim=-100)
    ax.legend(loc='upper left', bbox_to_anchor=(0.95, 0.95))
    # Customize the number of ticks on each axis
    ax.xaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    ax.yaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    ax.zaxis.set_major_locator(LinearLocator(5))  # Set 5 evenly spaced ticks
    # Show the plot
    plt.tight_layout()
    plt.savefig('./Q3_plots/gooding_orbits.png')
    plt.show()


def save_as_rso_obj(epoch, state, std_pos, std_vel):
    tudat_time = time_conversion.date_time_from_epoch(epoch)
    microsec = (tudat_time.seconds - int(tudat_time.seconds))*1000000
    utc = datetime(year=tudat_time.year, month=tudat_time.month, day=tudat_time.day, hour=tudat_time.hour,
                            minute=tudat_time.minute, second=int(tudat_time.seconds), microsecond=int(microsec))
    covar = np.diag(np.array([std_pos**2, std_vel**2]))
    rso_dict = {'UTC': utc, 'state': state.reshape(6,1), 'covar': covar, 'mass': 10, 'area': 1, 'Cd': 1.2, 'Cr': 1.2}

    rso = {98765: rso_dict}

    with open('./Q3_plots/rso_estimated.pkl', 'wb') as f:
        pickle.dump(rso, f)
