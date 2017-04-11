import numpy as np
import trackpy as tp
import pandas as pd
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

font = {'family':'normal', 'size':14}
mpl.rc('font', **font)
mpl.rc('text', usetex=True)

def np_round(num):
    return np.round(num, decimals=3)

def whitepane(ax):
    ''' Sets the pane color to white'''
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

def produceLine(x,y, color):
    ''' Provides segments generated from points x,y'''
    n = len(x)
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)
    lc = LineCollection(segments, cmap=color, linewidth = 1.5,
                        norm = plt.Normalize(0,n))
    lc.set_array(x)
    lc.set_linewidth = 5
    return lc

def msd_analysis(fit_x, fit_y):
    '''Performs trackpy analysis and fitting to MSD data.'''
    df = pd.DataFrame({'frame':range(len(fit_x)), 'x': fit_x, 
                       'y': fit_y})

    result = tp.msd(df, 0.135, 30, detail=True)
    t = np.arange(100)*1./30
    coeffs = np.polyfit(t, result['msd'], 2)
    poly = coeffs[0]*t**2 + coeffs[1]*t + coeffs[2]

    return result['msd'], poly, t

def main(traj_x, traj_y, fit, cascade_name):
    '''
    main produces a profile summary plot depicting the trajectory,
    the estimated trajectory, the distribution of times spent
    on feature detection, an MSD plot and an error plot.
    '''

    # Necessary Parameters.
    
    # Calculate necessary quanties.
    fit_t = fit[:,0]
    fit_x = fit[:,1]
    fit_y = fit[:,2]

    # Calculate error.
    error_x = traj_x - fit_x
    error_y = traj_y - fit_y

    # Make the 5 plots.
    fig = plt.figure(facecolor='w', figsize=(16,10))
    fig.suptitle('Profile Summary without Error Correction', fontsize=20)
    gs = gridspec.GridSpec(3,5)
    gs.update(left = 0.05, bottom = 0.05, wspace = 0.4, hspace = 0.3)
    
    # Make the trajectory plot.
    left =  np.min(traj_x)*0.9
    right = np.max(traj_x)*1.1
    up =    np.max(traj_y)*1.1
    down =  np.min(traj_y)*0.9
    skip = 5
    lc1 = produceLine(traj_x[::skip], traj_y[::skip], 'summer')
    lc2 = produceLine(fit_x[::skip], fit_y[::skip], 'winter')
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax1.add_collection(lc1)
    ax1.add_collection(lc2)
    ax1.set_xlim([left, right])
    ax1.set_ylim([down, up])
    ax1.set_xlabel('x [pix]')
    ax1.set_ylabel('y [pix]')
    ax1.legend([lc1,lc2],["Trajectory", "Fit"])
    
    # Make the MSD plot.
    msd, poly, t = msd_analysis(traj_x, traj_y)
    ax2 = fig.add_subplot(gs[0,2])
    lc1 = produceLine(t, msd, 'summer')
    lc2 = produceLine(t, poly, 'winter')
    ax2.add_collection(lc1)
    ax2.add_collection(lc2)
    ax2.set_xlabel(r'$\tau$ [s]')
    ax2.set_ylabel(r"$\Delta r^2(\tau)$")
    ax2.set_xlim([0, 3])
    ax2.set_ylim([0, 3])

    # Make the Time probability plot.
    ax3 = fig.add_subplot(gs[1,2])
    n, bins, patches = ax3.hist(fit_t*1000, 50, normed = 1, facecolor = 'red', alpha = 0.75)
    ax3.axis([9, 12, 0, 5])
    ax3.set_ylabel(r"P($\Delta t$)")
    ax3.set_xlabel(r"$\Delta t$ [ms]")
    ax3.xaxis.set_ticks([9, 10, 11, 12])

    # Make the error plot.
    xy = np.vstack([error_x, error_y])
    c = gaussian_kde(xy)(xy)*.3
    ax4 = fig.add_subplot(gs[0:2, 3:])
    ax4.scatter(error_x, error_y, c=c, s=100, edgecolor='', alpha=.7, facecolors='none')
    ax4.plot([-10,10],[0,0], '--', color = 'black', lw=2)
    ax4.plot([0,0], [-10,10], '--', color = 'black', lw=2)
    ax4.set_xlabel(r'$\Delta X$ error [pix]')
    ax4.set_ylabel(r'$\Delta Y$ error [pix]')
    ax4.set_xlim([-6,6])
    ax4.set_ylim([-6,6])
    
    # Annotations.
    ax5 = fig.add_subplot(gs[2,:])
    ax5.axis('off')
    sigma = np.sqrt(poly[0])
    avg_time = np.mean(fit_t)*1000
    avg_errx, avg_erry =  np.mean(error_x), np.mean(error_y)
    std_errx, std_erry = np.std(error_x), np.std(error_y)
    sigma, avg_time, avg_errx, avg_erry, std_errx, std_erry = map(np_round, [sigma, avg_time, avg_errx, avg_erry, std_errx, std_erry])

    ann_fs = 16
    ax5.annotate('Cascade classifier: {}'.format(cascade_name), xy=(0,0), xytext=(0., .8), fontsize=ann_fs)
    ax5.annotate(r'\underline{Time Analysis}', fontsize=ann_fs, xy=(0,0), xytext=(0.,.55))
    ax5.annotate(r'Average total detection time:  {} ms'.format(avg_time), xy=(0,0), xytext=(.0, .4), fontsize=ann_fs)
    ax5.annotate(r'\underline{Spatial Resolution}', fontsize=ann_fs, xy=(0,0), xytext=(.6,.55))
    ax5.annotate(r'$\Delta X$:  {} $\pm$ {} pix'.format(avg_errx, std_errx), xy=(0,0), xytext=(.6, .4), fontsize=ann_fs)
    ax5.annotate(r'$\Delta Y$:  {} $\pm$ {} pix'.format(avg_erry, std_errx), xy=(0,0), xytext=(.6, .25), fontsize=ann_fs)
    ax5.annotate(r'Static Localization error $\sigma_s$: {} pix'.format(sigma), xy=(0,0), xytext=(.6, .1), fontsize=ann_fs)
    


    # Plot the result.
    plt.show()
     
def load_fit(fname):
    fit_data = np.loadtxt(fname)
    return fit_data

def load_traj(fname):
    traj_data = np.loadtxt(fname)
    traj_data = traj_data.flatten()
    n_samples = len(traj_data)/2
    traj_x = traj_data[:n_samples]
    traj_y = traj_data[n_samples:]
    traj_y = 479 - traj_y
    return traj_x, traj_y

if __name__ == '__main__':
    # Begin argparse.
    import argparse
    parser = argparse.ArgumentParser()

    # Set default args.
    root = '../test_imgs/diffusion/'
    parser.add_argument('-name', type=str, help='Name of the cascade.', 
                        default='../cascade_example.xml')
    parser.add_argument('-traj_name', type=str, 
                        help='Name of the trajectory file.', 
                        default=root + 'trajectory.dat')
    parser.add_argument('-fit_name', type=str,
                        help='Name of the fit data file', 
                        default=root + 'fit_result.dat')
    
    args = parser.parse_args()

    # Perform profile summary.
    fit_data = load_fit(args.fit_name)
    traj_x, traj_y = load_traj(args.traj_name)
    main(traj_x, traj_y, fit_data, args.name)
