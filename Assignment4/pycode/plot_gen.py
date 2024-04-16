import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tudatpy import constants


"""Module with reusable plotting functions for generic plotting and single-use plotting functions for the specific project."""



def fig_formatter(ax: plt.axes, opts: dict):
    """Configures axes objs according to the options specified in a dictionary. Parameters that can be
    configured: 
        - 'title' (str), 'title_fontsize' (int);
        - 'xlabel' (str), 'ylabel' (str), 'label_fontsize' (int), 'xscale' (str), 'yscale (str),
        'xlim' (list), 'ylim' (list), 'tick_fontsize' (int);
        - 'legend' (bool), 'legend_fontsize' (int), 'legend_location' (str);
        - 'grid' (bool);

    Arguments:
        ax: axis matplotlib object.
        opts: dictionary with the configuration options.
    """

    # Check which options are specified in the dictionary and need to be set
    if 'label_fontsize' in opts:
        label_fs = opts['label_fontsize']
    else:
        label_fs = 14
    if 'title_fontsize' in opts:
        title_fs = opts['title_fontsize']
    else:
        title_fs = 16
    if 'legend_fontsize' in opts:
        legend_fs = opts['legend_fontsize']
    else:
        legend_fs = 14
    if 'legend_location' in opts:
        legend_loc = opts['legend_location']
    else:
        legend_loc = 'upper right'
    if 'xlabel' in opts:
        ax.set_xlabel(opts['xlabel'], fontsize=label_fs)
    if 'ylabel' in opts:
        ax.set_ylabel(opts['ylabel'], fontsize=label_fs)
    if 'zlabel' in opts:
        ax.set_zlabel(opts['zlabel'], fontsize=label_fs)
    if 'xlim' in opts:
        ax.set_xlim(opts['xlim'])
    if 'ylim' in opts:
        ax.set_ylim(opts['ylim'])
    if 'grid' in opts:
        ax.grid(opts['grid'])
    if 'title' in opts:
        ax.set_title(opts['title'],fontsize=title_fs)
    if 'tick_fontsize' in opts:
        ax.tick_params(axis='both',which='major',labelsize=opts['tick_fontsize'])
    if 'legend' in opts and opts['legend'] is True:
        ax.legend(fontsize=legend_fs, loc=legend_loc)
    if 'xscale' in opts:
        ax.set_xscale(opts['xscale'])
    if 'yscale' in opts:
        ax.set_yscale(opts['yscale'])

    return


def plotter_2d_3plot(ax: plt.axes, data1: np.ndarray[float], data2: np.ndarray[float], data3: np.ndarray[float],
                      data4: np.ndarray[float], param_dict1: dict, param_dict2: dict, param_dict3: dict):
    """Plots three different arrays of data on the same 2D plot. The options for each array, such
    as 'color' or 'label', can be configured in dictionaries.

    Arguments:
        ax: matplotlib axis object.
        data1: array of data for x axis.
        data2: first array of data for y axis.
        data3: second array of data for y axis.
        data4: third array of data for y axis.
        param_dict1: parameters dictionary for first array.
        param_dict2: parameters dictionary for second array.
        param_dict3: parameters dictionary for third array.
    """

    # Plot the data
    out = ax.plot(data1, data2, **param_dict1)
    out = ax.plot(data1, data3, **param_dict2)
    out = ax.plot(data1, data4, **param_dict3)
    return


def plotter_2d_2plot(ax: plt.axes, data1: np.ndarray[float], data2: np.ndarray[float], data3: np.ndarray[float],
                     param_dict1: dict, param_dict2: dict):
    """Plots two different arrays of data on the same 2D plot. The options for each array, such
    as 'color' or 'label', can be configured in dictionaries.

    Arguments:
        ax: matplotlib axis object.
        data1: array of data for x axis.
        data2: first array of data for y axis.
        data3: second array of data for y axis.
        param_dict1: parameters dictionary for first array.
        param_dict2: parameters dictionary for second array.
    """

    # Plot the data
    out = ax.plot(data1, data2, **param_dict1)
    out = ax.plot(data1, data3, **param_dict2)
    return


def plotter_2d_1plot(ax: plt.axes, data1: np.ndarray[float], data2: np.ndarray[float], param_dict1: dict):
    """Plots an array of data on a 2D plot. The options for the array, such
    as 'color' or 'label', can be configured in a dictionary.

    Arguments:
        ax: matplotlib axis object.
        data1: array of data for x axis.
        data2: array of data for y axis.
        param_dict1: parameters dictionary.
    """

    # Plot the data
    out = ax.plot(data1, data2, **param_dict1)
    return


def cart_plot(t: np.ndarray[float], r: np.ndarray[float], v: np.ndarray[float]) -> plt.figure:
    """Plots the cartesian components along x,y and z of the orbital radius and velocity of a satellite
    in two subplots side by side.

    Arguments:
        t: time serie in seconds.
        r: radii in km.
        v: velocities in km/s
    Returns:
        fig_cart: figure with the requested plots.
    """

    # Define the writing sizes
    tick_fs = 12
    label_fs = 12
    title_fs = 12
    legend_fs = label_fs

    # Create the 1x2 figure
    fig_cart,axs = plt.subplots(1,2,sharex = True,figsize=[9,9],layout='constrained')
    # fig_cart.suptitle('Cartesian coordinates',fontsize=22)
    # Define the radius subplot
    ax_r = axs[0] 
    param_dict_rx = {
        'label':'Position along x', 
        'color':'red'
    }
    param_dict_ry = {
        'label':'Position along y', 
        'color':'blue'
    }
    param_dict_rz = {
        'label':'Position along z', 
        'color':'green'
    }
    opts_r = {
        'xlabel':'t [hr]',
        'ylabel':'Position [km]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        'title':'Position',
        'title_fontsize':title_fs,
        'legend':True,
        'legend_fontsize':legend_fs,
        'legend_location':'upper right',
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_3plot(ax_r,t,r[:,0],r[:,1],r[:,2],param_dict_rx,param_dict_ry, param_dict_rz)
    fig_formatter(ax_r,opts_r)
    # Define the velocity subplot
    ax_v = axs[1] 
    param_dict_vx = {
        'label':'Velocity along x', 
        'color':'red'
    }
    param_dict_vy = {
        'label':'Velocity along y', 
        'color':'blue'
    }
    param_dict_vz = {
        'label':'Velocity along z', 
        'color':'green'
    }
    opts_v = {
        'xlabel':'t [hr]',
        'ylabel':'Velocity [km/s]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        'title':'Velocity',
        'title_fontsize':title_fs,
        'legend':True,
        'legend_fontsize':legend_fs,
        'legend_location':'upper right',
        'grid':True
    }
    plotter_2d_3plot(ax_v,t,v[:,0],v[:,1],v[:,2],param_dict_vx,param_dict_vy, param_dict_vz)
    fig_formatter(ax_v,opts_v)

    return fig_cart


def kep_plot(t: np.ndarray[float], kepler_elements: np.ndarray[float]) -> plt.figure:
    """Plots the 6 keplerian elements of a satellite in six separate plots.

    Arguments:
        t: time serie in days.
        kepler_elements: keplerian elements in m and rad.
    Returns:
        fig_kep: figure with the requested plots.
    """

    a = kepler_elements[:,0] / 1000  # Conversion to km
    e = kepler_elements[:,1]
    i = kepler_elements[:,2]
    arg_per = kepler_elements[:,3]
    raan = kepler_elements[:,4]
    theta = kepler_elements[:,5]

    # Conversions rad -> deg
    i = np.rad2deg(i)
    theta = np.rad2deg(theta)
    raan = np.rad2deg(raan)
    arg_per = np.rad2deg(arg_per)

    # Define the writing sizes
    tick_fs = 12
    label_fs = 12
    title_fs = 12

    # Create the 3x2 figure
    fig_kep,axs = plt.subplots(3,2,sharex = True,figsize=[9,9],layout='constrained')
    # fig_kep.suptitle('Keplerian coordinates',fontsize=22)
    # Define the semi-major axis subplot
    ax_a = axs[0,0]
    param_dict_a = {
        'label':'Semi-major axis', 
        'color':'red'
    }
    opts_a = {
        # 'xlabel':'time [hr]',
        'ylabel':'a [km]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        # 'title':'Semi-major axis',
        # 'title_fontsize':title_fs,
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_1plot(ax_a,t,a,param_dict_a)
    fig_formatter(ax_a,opts_a)
    # Define the true anomaly subplot
    ax_theta = axs[1,0]
    param_dict_theta = {
        'label':'True anomaly', 
        'color':'blue'
    }
    opts_theta = {
        # 'xlabel':'time [hr]',
        'ylabel':r'$\theta$ [deg]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        # 'title':'True anomaly',
        # 'title_fontsize':title_fs,
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_1plot(ax_theta,t,theta,param_dict_theta)
    fig_formatter(ax_theta,opts_theta)
    # Define the eccentricity subplot
    ax_e = axs[2,0]
    param_dict_e = {
        'label':'Eccentricity', 
        'color':'green'
    }
    opts_e = {
        'xlabel':'t [days]',
        'ylabel':'e [-]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        # 'title':'Eccentricity',
        # 'title_fontsize':title_fs,
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_1plot(ax_e,t,e,param_dict_e)
    fig_formatter(ax_e,opts_e)
    # Define the inclination subplot
    ax_i = axs[0,1]
    param_dict_i = {
        'label':'Inclination', 
        'color':'olive'
    }
    opts_i = {
        # 'xlabel':'time [hr]',
        'ylabel':'i [deg]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        # 'title':'Inclination',
        # 'title_fontsize':title_fs,
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_1plot(ax_i,t,i,param_dict_i)
    fig_formatter(ax_i,opts_i)
    # Define the right ascension of the ascending node subplot
    ax_raan = axs[1,1]
    param_dict_raan = {
        'label':'RAAN', 
        'color':'indigo'
    }
    opts_raan = {
        # 'xlabel':'time [hr]',
        'ylabel':r'$\Omega$ [deg]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        # 'title':'Right Ascension of the Ascending Node',
        # 'title_fontsize':title_fs,
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_1plot(ax_raan,t,raan,param_dict_raan)
    fig_formatter(ax_raan,opts_raan)
    # Define the argument of periapsis subplot
    ax_arg_per = axs[2,1]
    param_dict_arg_per = {
        'label':'Argument of periapsis', 
        'color':'maroon'
    }
    opts_arg_per = {
        'xlabel':'t [days]',
        'ylabel':r'$\omega$ [deg]',
        'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        # 'title':'Argument of periapsis',
        # 'title_fontsize':title_fs,
        'grid':True,
        # 'ylim':[-180,180],
        'tick_fontsize':tick_fs,

    }
    plotter_2d_1plot(ax_arg_per,t,arg_per,param_dict_arg_per)
    fig_formatter(ax_arg_per,opts_arg_per)

    return fig_kep


def generic_plot(t:np.ndarray[float], fun: np.ndarray[float]) -> plt.figure:
    """Generic 2D plot.

    Arguments:
        t: time serie of the propagation.
        fun: magnitude of perturbating acceleration.
    Returns:
        fig: figure with the requested plots.
    """

    # Define the writing sizes
    tick_fs = 12
    label_fs = 14
    title_fs = 16
    legend_fs = label_fs

    # Create the figure
    fig,ax = plt.subplots(figsize=[9,6])
    # fig.suptitle('Sun acceleration magnitude',fontsize=22)
    # Define the position residuals plot
    param_dict_acc = {
        'color':'red',
        # 'label':r'$\rho$'
    }
    opts_acc = {
        'xlabel':'?',
        'ylabel':'?',
        # 'xlim':[t[0],t[-1]],
        'label_fontsize':label_fs,
        'title_fontsize':title_fs,
        'grid':True,
        'tick_fontsize':tick_fs,
    }
    plotter_2d_1plot(ax,t,fun,param_dict_acc)
    fig_formatter(ax,opts_acc)
    plt.tight_layout()

    return fig

def orbit_a3(r: np.ndarray[float]):
    """3D plotter for orbits to be used during debug. It receives an array of positions and
    displays them in 3D space.

    Arguments:
        r: array of positions in km.
    """

    # Define the writing sizes
    tick_fs = 12
    label_fs = 14
    title_fs = 16
    legend_fs = label_fs

    # Definition
    X = r[:,0] / 10**9
    Y = r[:,1] / 10**9
    Z = r[:,2] / 10**9

    # Plot and show

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    param_dict = {
        'color':'blue',
        # 'label':r'$\rho$'
    }
    opts = {
        'xlabel':r'x [$10^6$ km]',
        'ylabel':r'y [$10^6$ km]',
        'zlabel':r'z [$10^6$ km]',
        'label_fontsize':label_fs,
        'title_fontsize':title_fs,
        'legend':True,
        'grid':True,
        'tick_fontsize':tick_fs,
    }

    ax.plot(X, Y, Z, label='3D orbit',**param_dict)

    # Equalise axes
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.plot(0,0,0,'ro',label= 'Sun') 

    fig_formatter(ax,opts)
    plt.tight_layout()

    return fig

def orbit_generic(r: np.ndarray[float]):
    """3D plotter for orbits to be used during debug. It receives an array of positions and
    displays them in 3D space.

    Arguments:
        r: array of positions in m.
    """

    # Define the writing sizes
    tick_fs = 12
    label_fs = 14
    title_fs = 16
    legend_fs = label_fs

    # Definition
    X = r[:,0]
    Y = r[:,1]
    Z = r[:,2]

    # Plot and show

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    param_dict = {
        'color':'blue',
        # 'label':r'$\rho$'
    }
    opts = {
        'xlabel':r'x [m]',
        'ylabel':r'y [m]',
        'zlabel':r'z [m]',
        'label_fontsize':label_fs,
        'title_fontsize':title_fs,
        'legend':True,
        'grid':True,
        'tick_fontsize':tick_fs,
    }

    ax.plot(X, Y, Z, label='3D orbit',**param_dict)

    # Equalise axes
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.plot(0,0,0,'ro') 

    fig_formatter(ax,opts)
    plt.tight_layout()

    return fig



