import pycode.ConjunctionUtilities as cutils
from tudatpy.astro.time_conversion import epoch_from_date_time_components
from tudatpy import constants
import pycode.TudatPropagator_q3 as prop
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

if __name__ == "__main__":
    filename = 'data/group2/estimated_rso_catalog.pkl'
    rso_dict = cutils.read_catalog_file(filename)
    norad_ids = [91509, 91686, 91332, 91395, 91883, 40940]
    # norad_ids = list(rso_dict.keys())
    # norad_ids = [40940]
    bodies_to_create = ["Earth", "Sun", "Moon"]

    location_groundstation_lon = 132.951944
    location_groundstation_lat = -12.432778
    R = 6378000
    h = 42165000

    Lambda = np.arccos((R) / (R + h))
    Phi_E = np.linspace(0, 2 * np.pi, num=5000)

    # Create hemisphere function
    mask_E = []
    for i in range(len(Phi_E)):
        val = (-Phi_E[i]) % 2 * np.pi
        if val >= 0 and val < np.pi:
            mask_E.append(1.0)
        else:
            mask_E.append(-1.0)

    # Calculate horizon coordinates on the map.
    colat_horizon = np.arccos(
        np.cos(Lambda) * np.cos((90 - location_groundstation_lat) / 180 * np.pi) + np.sin(Lambda) * np.sin(
            (90 - location_groundstation_lat) / 180 * np.pi) * np.cos(Phi_E % 2 * np.pi))
    DL = ((mask_E * np.arccos(
        (np.cos(Lambda) - np.cos(colat_horizon) * np.cos((90 - location_groundstation_lat) / 180 * np.pi)) / (
                    np.sin((90 - location_groundstation_lat) / 180 * np.pi) * np.sin(colat_horizon)))))

    LAT_horizon = (90 - (colat_horizon / np.pi * 180))
    LON_horizon_abs = ((location_groundstation_lon / 180 * np.pi - DL) / np.pi * 180)
    LON_horizon = np.where(LON_horizon_abs <= 180, LON_horizon_abs, LON_horizon_abs - 360)

    int_params = {'tudat_integrator': 'rkf78', 'step': 10., 'max_step': 1000., 'min_step': 1e-3, 'rtol': 1e-12,
                  'atol': 1e-12}

    dic_lat = {}
    dic_lon = {}

    for norad_id in norad_ids:
        print(f"RSO : {norad_id}")
        obj = rso_dict[norad_id]
        X0 = obj['state']
        utc = obj['UTC']
        t0 = epoch_from_date_time_components(
            utc.year, utc.month, utc.day, utc.hour, utc.minute, float(utc.second)
        )
        tspan = 1 * constants.JULIAN_DAY
        trange = np.array([t0, t0 + tspan])
        state_params = {}
        state_params['Cd'] = obj['Cd']
        state_params['Cr'] = obj['Cr']
        state_params['mass'] = obj['mass']
        state_params['area'] = obj['area']
        state_params['sph_deg'] = 8
        state_params['sph_ord'] = 8
        state_params['central_bodies'] = ['Earth']
        state_params['bodies_to_create'] = bodies_to_create

        tout, xout, latitudes, longitudes, shadow = prop.propagate_orbit(X0, trange, state_params, int_params, latlon_flag=True)
        shadow_index = np.where(shadow < 1)[0]
        shadow_times = (tout[np.where(shadow < 1)] - t0)/3600
        dic_lat[norad_id] = latitudes
        dic_lon[norad_id] = longitudes
        try:
            tin_shadow = (tout[shadow_index[0] - 1] - t0) / 3600  # Last time in full sun before entering the shadow
            tout_shadow = (tout[shadow_index[-1] + 1] - t0) / 3600  # First time in full sun after leaving the shadow
            print(f'Shadow time: {tin_shadow} <= t <= {tout_shadow}')
        except IndexError:
            print(f'Shadow time: none')
            pass

        # latitudes = latitudes[np.where(shadow == 1)]
        # longitudes = longitudes[np.where(shadow == 1)]
        print(f"Shadow times: {shadow_times}")
        print("------------------------------")

    plt.figure(figsize=(12, 8), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    for norad_id in norad_ids:
        longitudes = dic_lon[norad_id]
        latitudes = dic_lat[norad_id]
        if norad_id == 40940:
            ax.scatter(longitudes / np.pi * 180, latitudes / np.pi * 180, label=norad_id, marker='*', s=75)
        else:
            ax.scatter(longitudes / np.pi * 180, latitudes / np.pi * 180, label=norad_id, marker='o', s=20)

        sin_rho = R / (R + h)
        DL_e = np.deg2rad(np.rad2deg(longitudes) - location_groundstation_lon)
        Lambda_e = np.arccos(
            np.cos(latitudes) * np.cos(np.deg2rad(location_groundstation_lat)) + np.sin(latitudes) * np.sin(
                np.deg2rad(location_groundstation_lat)) * np.cos(DL_e))

        eta = np.arctan2(sin_rho * np.sin(Lambda_e), 1 - sin_rho * np.cos(Lambda_e))
        elevation_abs = np.rad2deg(np.arccos(np.sin(eta) / sin_rho))
        elevation_lambda_check = np.where(Lambda_e <= np.arccos(R / (R + h)), elevation_abs, 0)
        elevation = np.where(np.abs(DL_e) <= 0.5 * np.pi, elevation_lambda_check, 0)

        # Plot elevation
        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.set_title(f'Elevation of object {norad_id}')
        ax2.plot((tout - t0) / 3600, elevation, color='red')
        ax2.set_xlabel('Time [hours since beginning epoch]')
        ax2.set_ylabel('Elevation [deg]')
        plt.grid()
        plt.show()

    ax.scatter(location_groundstation_lon, location_groundstation_lat, color='red', marker='*', s=100, label="Sensor")
    # ax.scatter(LON_horizon, LAT_horizon, color='red', marker='.', s=5)
    ax.gridlines(draw_labels=True)
    ax.legend()
    plt.show()
