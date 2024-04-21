from tudatpy.astro.time_conversion import julian_day_to_calendar_date
from tudatpy import constants
import pandas as pd
import os
import inspect

# Define path to objects datafile
current_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
src_dir = os.path.dirname(current_dir)
matlab_outputs_dir = os.path.join(src_dir, 'CARA_matlab', 'MonteCarloPc', 'outputs')
output_dir = os.path.join(src_dir, 'output', 'CDM')

def write_cdm(epoch, tca, d_euc, d_mahal, speed, rel_pos_ric, Pc, norad_rso, X_sat, X_rso, P_sat, P_rso):
    """
    Writes CDM
    :param P_rso: RSO covariance matrix 6x6
    :param P_sat: Satellite covariance matrix 6x6
    :param X_rso: RSO state vector 6x1
    :param X_sat: Satellite state vector 6x1
    :param norad_rso: Norad ID rso
    :param Pc: Foster Probability of collision
    :param rel_pos_ric: Relative position in RIC reference frame
    :param speed: Relative speed
    :param d_euc: Euclidean distance at TCA
    :param d_mahal: Mahalanobis distance at TCA
    :param tca: TCA
    :param epoch: Date of creation file (equivalent to RSO epoch)
    :return: filename
    """

    '''
    Message shall contain:
    - HEADER:
        - creation date (Epoch)
    - RELATIVE DATA/METADATA:
        - TCA
        - Miss distance (euclidean)
        - Relative Speed 
        - Relative position rsw
        - Probability of impact and method
        - Mahalanobis distance
    - METADATA
        - object ( OBJECT 1 or OBJECT 2)
        - NORAD id (called object designator)
        - catalog name
        - object name 
        - MANEUVERABLE (yes/no/NA)
        - Ref_frame
    - DATA
        - x, y, z
        - covariance        
    '''
    date = epoch.strftime("%Y-%m-%d T%H:%M:%S")
    header = [
        {"Column1": "Creation Date", "Column2": date, "Column3": ""}
    ]
    jd = 2451545.0      # J2000
    tca_abs = tca/constants.JULIAN_DAY + jd
    tca_date = julian_day_to_calendar_date(tca_abs).strftime("%Y-%m-%d T%H:%M:%S")
    r = rel_pos_ric[0][0]
    i = rel_pos_ric[1][0]
    c = rel_pos_ric[2][0]
    direct = os.path.join(matlab_outputs_dir, str(norad_rso))

    # if Pc > 1e-6:
    # if Pc < 1e-6, CARA method does not calculate MC probability

    P_foster = Pc
    # FInd better Foster estimate (if present)
    foster_path = os.path.join(direct, 'Pc_foster.dat')
    if os.path.isfile(foster_path):
        file = pd.read_csv(foster_path, header=None)
        P_foster = file[0][0]

    mc_flag = False
    mc_path = os.path.join(direct, 'Pc_MonteCarlo.dat')
    if os.path.isfile(mc_path):
        mc_flag = True
        print('CDM Mc funziona')
        file = pd.read_csv(mc_path, header=None)
        P_mc = file[0][0]

    if mc_flag:
        relative = [
            {"Column1": "TCA", "Column2": tca_date, "Column3": ""},
            {"Column1": "Miss Distance", "Column2": d_euc, "Column3": '[m]'},
            {"Column1": "Relative Speed", "Column2": speed, "Column3": '[m/s]'},
            {"Column1": "Relative Position R", "Column2": r, "Column3": '[m]'},
            {"Column1": "Relative Position I", "Column2": i, "Column3": '[m]'},
            {"Column1": "Relative Position C", "Column2": c, "Column3": '[m]'},
            {"Column1": "Collision Probability", "Column2": P_foster, "Column3": ''},
            {"Column1": "Collision Probability method", "Column2": 'Foster', "Column3": ''},
            {"Column1": "Collision Probability", "Column2": P_mc, "Column3": ''},
            {"Column1": "Collision Probability method", "Column2": 'Monte Carlo', "Column3": ''},
            {"Column1": "Mahalanobis Distance", "Column2": d_mahal, "Column3": ''}
        ]
    else:
        relative = [
            {"Column1": "TCA", "Column2": tca_date, "Column3": ""},
            {"Column1": "Miss Distance", "Column2": d_euc, "Column3": '[m]'},
            {"Column1": "Relative Speed", "Column2": speed, "Column3": '[m/s]'},
            {"Column1": "Relative Position R", "Column2": r, "Column3": '[m]'},
            {"Column1": "Relative Position I", "Column2": i, "Column3": '[m]'},
            {"Column1": "Relative Position C", "Column2": c, "Column3": '[m]'},
            {"Column1": "Collision Probability", "Column2": P_foster, "Column3": ''},
            {"Column1": "Collision Probability method", "Column2": 'Foster', "Column3": ''},
            {"Column1": "Mahalanobis Distance", "Column2": d_mahal, "Column3": ''}
        ]


    norad_sat = 40920
    metadata_sat = [
        {"Column1": "Object", "Column2": 'Satellite', "Column3": ''},
        {"Column1": "Norad ID", "Column2": norad_sat, "Column3": ''},
        {"Column1": "Object Name", "Column2": 'Sky Muster', "Column3": ''},
        {"Column1": "Covariance Method", "Column2": 'Calculated', "Column3": ''},
        {"Column1": "Maneuverable", "Column2": 'Yes', "Column3": ''},
        {"Column1": "Reference Frame", "Column2": 'ECI', "Column3": ''}
    ]
    metadata_rso = [
        {"Column1": "Object", "Column2": 'Resident Space Object', "Column3": ''},
        {"Column1": "Norad ID", "Column2": norad_rso, "Column3": ''},
        {"Column1": "Covariance Method", "Column2": 'Calculated', "Column3": ''},
        {"Column1": "Maneuverable", "Column2": 'N/A', "Column3": ''},
        {"Column1": "Reference Frame", "Column2": 'ECI', "Column3": ''}
    ]
    x_sat = X_sat[0][0]
    y_sat = X_sat[1][0]
    z_sat = X_sat[2][0]
    vx_sat = X_sat[3][0]
    vy_sat = X_sat[4][0]
    vz_sat = X_sat[5][0]
    x_rso = X_rso[0][0]
    y_rso = X_rso[1][0]
    z_rso = X_rso[2][0]
    vx_rso = X_rso[3][0]
    vy_rso = X_rso[4][0]
    vz_rso = X_rso[5][0]
    data_sat = [
        {"Column1": "X", "Column2": x_sat / 1000, "Column3": '[km]'},
        {"Column1": "Y", "Column2": y_sat / 1000, "Column3": '[km]'},
        {"Column1": "Z", "Column2": z_sat / 1000, "Column3": '[km]'},
        {"Column1": "X_dot", "Column2": vx_sat / 1000, "Column3": '[km/s]'},
        {"Column1": "Y_dot", "Column2": vy_sat / 1000, "Column3": '[km/s]'},
        {"Column1": "Z_dot", "Column2": vz_sat / 1000, "Column3": '[km/s]'},
        {"Column1": "Cx_x", "Column2": P_sat[0, 0], "Column3": '[m**2]'},
        {"Column1": "Cy_x", "Column2": P_sat[0, 1], "Column3": '[m**2]'},
        {"Column1": "Cy_y", "Column2": P_sat[1, 1], "Column3": '[m**2]'},
        {"Column1": "Cz_x", "Column2": P_sat[0, 2], "Column3": '[m**2]'},
        {"Column1": "Cz_y", "Column2": P_sat[1, 2], "Column3": '[m**2]'},
        {"Column1": "Cz_z", "Column2": P_sat[2, 2], "Column3": '[m**2]'},
        {"Column1": "Cxdot_x", "Column2": P_sat[3, 0], "Column3": '[m**2/s]'},
        {"Column1": "Cxdot_y", "Column2": P_sat[3, 1], "Column3": '[m**2/s]'},
        {"Column1": "Cxdot_z", "Column2": P_sat[3, 2], "Column3": '[m**2/s]'},
        {"Column1": "Cxdot_xdot", "Column2": P_sat[3, 3], "Column3": '[m**2/s**2]'},
        {"Column1": "Cydot_x", "Column2": P_sat[4, 0], "Column3": '[m**2/s]'},
        {"Column1": "Cydot_y", "Column2": P_sat[4, 1], "Column3": '[m**2/s]'},
        {"Column1": "Cydot_z", "Column2": P_sat[4, 2], "Column3": '[m**2/s]'},
        {"Column1": "Cydot_xdot", "Column2": P_sat[4, 3], "Column3": '[m**2/s**2]'},
        {"Column1": "Cydot_ydot", "Column2": P_sat[4, 4], "Column3": '[m**2/s**2]'},
        {"Column1": "Czdot_x", "Column2": P_sat[5, 0], "Column3": '[m**2/s]'},
        {"Column1": "Czdot_y", "Column2": P_sat[5, 1], "Column3": '[m**2/s]'},
        {"Column1": "Czdot_z", "Column2": P_sat[5, 2], "Column3": '[m**2/s]'},
        {"Column1": "Czdot_xdot", "Column2": P_sat[5, 3], "Column3": '[m**2/s**2]'},
        {"Column1": "Czdot_ydot", "Column2": P_sat[5, 4], "Column3": '[m**2/s**2]'},
        {"Column1": "Czdot_zdot", "Column2": P_sat[5, 5], "Column3": '[m**2/s**2]'}
    ]

    data_rso = [
        {"Column1": "X", "Column2": x_rso / 1000, "Column3": '[km]'},
        {"Column1": "Y", "Column2": y_rso / 1000, "Column3": '[km]'},
        {"Column1": "Z", "Column2": z_rso / 1000, "Column3": '[km]'},
        {"Column1": "X_dot", "Column2": vx_rso / 1000, "Column3": '[km/s]'},
        {"Column1": "Y_dot", "Column2": vy_rso / 1000, "Column3": '[km/s]'},
        {"Column1": "Z_dot", "Column2": vz_rso / 1000, "Column3": '[km/s]'},
        {"Column1": "Cx_x", "Column2": P_rso[0, 0], "Column3": '[m**2]'},
        {"Column1": "Cy_x", "Column2": P_rso[0, 1], "Column3": '[m**2]'},
        {"Column1": "Cy_y", "Column2": P_rso[1, 1], "Column3": '[m**2]'},
        {"Column1": "Cz_x", "Column2": P_rso[0, 2], "Column3": '[m**2]'},
        {"Column1": "Cz_y", "Column2": P_rso[1, 2], "Column3": '[m**2]'},
        {"Column1": "Cz_z", "Column2": P_rso[2, 2], "Column3": '[m**2]'},
        {"Column1": "Cxdot_x", "Column2": P_rso[3, 0], "Column3": '[m**2/s]'},
        {"Column1": "Cxdot_y", "Column2": P_rso[3, 1], "Column3": '[m**2/s]'},
        {"Column1": "Cxdot_z", "Column2": P_rso[3, 2], "Column3": '[m**2/s]'},
        {"Column1": "Cxdot_xdot", "Column2": P_rso[3, 3], "Column3": '[m**2/s**2]'},
        {"Column1": "Cydot_x", "Column2": P_rso[4, 0], "Column3": '[m**2/s]'},
        {"Column1": "Cydot_y", "Column2": P_rso[4, 1], "Column3": '[m**2/s]'},
        {"Column1": "Cydot_z", "Column2": P_rso[4, 2], "Column3": '[m**2/s]'},
        {"Column1": "Cydot_xdot", "Column2": P_rso[4, 3], "Column3": '[m**2/s**2]'},
        {"Column1": "Cydot_ydot", "Column2": P_rso[4, 4], "Column3": '[m**2/s**2]'},
        {"Column1": "Czdot_x", "Column2": P_rso[5, 0], "Column3": '[m**2/s]'},
        {"Column1": "Czdot_y", "Column2": P_rso[5, 1], "Column3": '[m**2/s]'},
        {"Column1": "Czdot_z", "Column2": P_rso[5, 2], "Column3": '[m**2/s]'},
        {"Column1": "Czdot_xdot", "Column2": P_rso[5, 3], "Column3": '[m**2/s**2]'},
        {"Column1": "Czdot_ydot", "Column2": P_rso[5, 4], "Column3": '[m**2/s**2]'},
        {"Column1": "Czdot_zdot", "Column2": P_rso[5, 5], "Column3": '[m**2/s**2]'}
    ]
    filename = f'CDM_{norad_rso}.txt'
    file_path = os.path.join(output_dir, filename)
    max_col_width_1 = 35
    max_col_width_2 = 30
    with open(file_path, "w+") as file:

        file.write('CONJUNCTION DATA MESSAGE\n')

        for row in header:
            file.write(f"{row['Column1']:<{max_col_width_1}}{row['Column2']:<{max_col_width_2}}{row['Column3']}\n")

        file.write('\nRELATIVE DATA\n')
        for row in relative:
            col2 = f"{row['Column2']:.6g}" if isinstance(row['Column2'], float) else row['Column2']
            file.write(f"{row['Column1']:<{max_col_width_1}}{col2:<{max_col_width_2}}{row['Column3']}\n")

        file.write('\nOBJECTS DATA\n')
        for row in metadata_sat:
            col2 = f"{row['Column2']:.6g}" if isinstance(row['Column2'], float) else row['Column2']
            file.write(f"{row['Column1']:<{max_col_width_1}}{col2:<{max_col_width_2}}{row['Column3']}\n")

        for row in data_sat:
            col2 = f"{row['Column2']:.6g}" if isinstance(row['Column2'], float) else row['Column2']
            file.write(f"{row['Column1']:<{max_col_width_1}}{col2:<{max_col_width_2}}{row['Column3']}\n")

        file.write('\n')
        for row in metadata_rso:
            col2 = f"{row['Column2']:.6g}" if isinstance(row['Column2'], float) else row['Column2']
            file.write(f"{row['Column1']:<{max_col_width_1}}{col2:<{max_col_width_2}}{row['Column3']}\n")

        for row in data_rso:
            col2 = f"{row['Column2']:.6g}" if isinstance(row['Column2'], float) else row['Column2']
            file.write(f"{row['Column1']:<{max_col_width_1}}{col2:<{max_col_width_2}}{row['Column3']}\n")

    return file_path
