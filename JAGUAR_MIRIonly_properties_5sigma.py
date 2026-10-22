'''
Iris Jermann - PhD Student DAWN/DTU

This code: From JAGUAR simulation understand what are the properties of MIRI only.

The central function: extracts the
 
The second function: looks first if its QG or SFG, in case of SFR what the redshift is to look to 
which file to go to to extract the spectrum.

#IN THIS ONE I USE THE DEPTH FROM /Users/irisj/Documents/MIRIonly/everything_on_v7_pablo/v7_on_Pablo_img/depth 
update: use the depth in /Users/irisj/Documents/MIRIonly/everything_on_v7_pablo/step1_sextractor/D2_my_proper_depth/depth_sampled_iris_paper.csv
update oct 2024: use the depth in /Users/irisj/Documents/GitProjects/depth_code/depth_10nJy_in_midis.pkl

This just for the miri only: 
Then return one SED with the average SED and in the title the average properties and the limits
in the bands and the band passes, i.e. including mips and alma
and another SED plot where i plot them all 
update: skip the part where extracts the spectra

Or just make a file open data where i have the TC and all the spectra and combine all the spectra of the 
SFR and look per ID.

- THIS CODE MAKES A LOOP OVER ALL THE REALISATIONS
- THIS CODE TAKES RANDOM 4.7x4.7 arcmin AREA (TAKE SQUARE, NOT SHAPE OF THE FIELD LOL)
- THIS CODE ALSO DOES IT FOR > F444W SOURCESS : update : skip that part too

Outputs saved: /Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_snr/FourMicrons/
Update : outputs saved:  /Users/irisj/Documents/MIRIonly/JAGUAR_updated/
Update oct 2: /Users/irisj/Documents/MIRIonly/JAGUAR/

NOTE TO ME: FOR FLUXES ALWAYS USE 10nJy
OR MAGNITUDES AND THE ZP IS 28.9

Also, before running: activate camb

[08.04.2026.] This is the exact same code just is saves the color selection. 

'''


import matplotlib.pyplot as plt
#import camb

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import Counter

import astropy.cosmology as cosmo
from astropy.table import Table, vstack, hstack
#from hmf import MassFunction

# to calculate some integrals
from scipy.integrate import quad, cumtrapz

from astropy.io import fits
import astropy.units as u
from collections import Counter

import pandas as pd
#import pyphot
from numpy import random


#from pyphot import unit, Vega, Filter

show_plots = False
extract_spectra = False ## if True, then will extract all the spcectra, if False will not


def FindMIRIonlyAndPlot(Q_SF_tab,df_sel,R): #, median_MIRIony_nJy, F444W_2p5_nJy, F560W_2p5_nJy):
    '''
    Q_SF_tab: is the table with the fluxes for the 300000 objects for one realisation -> Q_SF_combined_MIRI_r1

    median_MIRIony_nJy: the median of the measured fluxes of the MIRI only in F560W
    F444W_2p5_nJy: the median flux of the x sources wiht 2.4<SNR<2.6 in F444W
    F560W_2p5_nJy: the same but it was only one source here

    returns: 
        HST_NIRCam_dark all the sources with fluxes below the 2.5 depth in HST and NIRCam band
        MIRIonly_potential the above + apply that they have to be brighter than 2.5 depth in MIRI
    '''

    ################### here is the HST NIRCam dark selection
    #df_sel['LIST'][:-1].values.tolist() # this is a list with the name of band HST, NIRCam
    Q_SF_combined_HST_NIRCam_fluxes = Q_SF_tab[df_sel['LIST'][:-1].values.tolist()] # this is all the measured fluxes in bands from HST and NIRCam
    np_depth = np.array([df_sel['1depth_nJy'][:-1].values.tolist()]* len(Q_SF_combined_HST_NIRCam_fluxes)) # here builds a matrix with the depth to compare to the matrix with fluxes, Q_SF_combined_HST_NIRCam_fluxes
    print(np_depth.shape)
    df_Q_SF = Q_SF_combined_HST_NIRCam_fluxes.to_pandas()

    # Get the NumPy array from the DataFrame
    np_Q_SF = df_Q_SF.values
    print(np_Q_SF.shape)

    np_bool_sel = np_depth < np_Q_SF ## put True where it is higher than depth, i.e. detected
    # this is a matrix containing True and False values comparing the matrix with the fluxes and the matrix with the depths

    # that way around so can just change in Q_SF_combined_MIRI_r1[df_sel['LIST'][:-1].values.tolist()] the [5:-1] to select other filters
    # and no need to change bool_sum == 0
    
    # bool_sum == 0 will keep the ones that have False everywhere meaning that there are all not smaller than the depth, i.e. they are non detected
    bool_sum = np.sum(np_bool_sel,axis=1)
    print(bool_sum)
    print(Counter(bool_sum))
    bool_ix_to_keep = bool_sum == 0 ## bool for each line, saying whether we keep it (i.e. if any in that line has a flux > depth)

    print('keep only the ones with sum == 0')
    print(np.where( bool_sum == 0 ))
    print(len(bool_ix_to_keep), np.sum(bool_ix_to_keep))

    HST_NIRCam_dark = Q_SF_tab[bool_ix_to_keep] ## now go back to the table with all the fluxes in all bands, not only HST and NIRcam anymore
    print('number of nircam hst dark:', len(HST_NIRCam_dark))

    ################### here is the MIRI selection
    # take the HST NIRCam dark and do bool miri with True when > 2.5 depth
    nc_depth = df_sel.loc[df_sel['BAND'] == 'f444w', '2.5depth_nJy'].iloc[0]
    midis_depth = df_sel.loc[df_sel['BAND'] == 'midis', '2.5depth_nJy'].iloc[0]
    # miriF560W_limit = df_sel['2.5depth_nJy'].iloc[-1] # miri threshold is the last number in there 
    bool_miri = HST_NIRCam_dark['F560W'] >= midis_depth # miriF560W_limit #*2/3 #/10  ## in nJy - correspond to a mag of 29.0
    MIRIonly_potential = HST_NIRCam_dark[bool_miri]
    print('number of miri only potential:', len(MIRIonly_potential))
    #exit()
    #### because code bugs when it is zero length, so at least it can make the plot
    if len(MIRIonly_potential) == 0:
        z, m, beta, sfr_10, sfr_100 = -99.0, -99.0, -99.0, -99.0, -99.0
    else:
        z = np.median(MIRIonly_potential['redshift'])
        m = np.median(MIRIonly_potential['mStar'])
        beta = np.median(MIRIonly_potential['beta'])
        sfr_10 = np.median(MIRIonly_potential['SFR_10'])
        sfr_100 = np.median(MIRIonly_potential['SFR_100'])
    

    fig = plt.figure(figsize=(8,6))
    # fig.suptitle('Average MIRI only properties from JAGUAR:\nz=%.2f ; %.3e M_star\nbeta=%.2f ; sfr_10=%.2f ; sfr_100=%.2f'%(z,m,beta,sfr_10,sfr_100))
    ax = fig.add_subplot(111)
    # ax.scatter( -0.024468854196301567*10, 1.0149720268006701 * 10, color='crimson' ,label = 'Median MIRI-only fluxes') ## bc in 10*nJy
    ### pour l'instant je vais juste mettre 5 sigma dans F560W v1.4 i.e., 2.42 nJy x 5
    # ax.scatter( 0 , 5*2.42, color='crimson' ,label = 'Median MIRI-only fluxes') ## bc in 10*nJy
    ax.scatter(HST_NIRCam_dark['NRC_F444W_fnu'], HST_NIRCam_dark['F560W'],  rasterized=True, alpha=0.5, s=15, label='HST & NIRCam dark (#%i)'%len(HST_NIRCam_dark))
    ax.scatter(MIRIonly_potential['NRC_F444W_fnu'], MIRIonly_potential['F560W'], label='MIRI only (%i candidates)'%len(MIRIonly_potential))
    ax.set_xlabel('Flux F444W [nJy]')
    ax.set_ylabel('Flux F560W [nJy]')
    #ax.axvline(3.162278  ,  color = 'green',label = 'F444W JADES depth') # 
    #ax.axhline(9.120108  ,  color = 'crimson',label = 'F560W MIDIS depth') #
    kk_ix = -4 # the index of the band we want on the x axis, here F444W and it is -4
    ax.axvline( nc_depth, ls='--',  color = 'green',label = r'F444W 1$\sigma$=%.2f nJy'%(nc_depth)) 
    ax.axhline( midis_depth , ls='--',  color = 'crimson',label = r'F560W 5$\sigma$=%.2f nJy'%(midis_depth)) 
    # ax.axvline(df_sel['2.5depth_nJy'].iloc[kk_ix], ls='--',  color = 'green',label = '%s 2.5 depth'%(df_sel['2.5depth_nJy'].iloc[kk_ix])) 
    # ax.axhline( df_sel['2.5depth_nJy'].iloc[-1] , ls='--',  color = 'crimson',label = '%s 2.5 depth'%(df_sel['2.5depth_nJy'].iloc[-1])) 
    # here in 10nJy and put the 2.5 i plug in down there for the selection in the code
    ax.legend()
    if show_plots == True:
        plt.show()
    else:
        None
    #exit()

    ############# plot the distributions of them all
    fig_dist = plt.figure(figsize=(8,8))
    # fig_dist.suptitle('Realisation %s'%R[-1])
    ax1, ax2, ax3 = fig_dist.add_subplot(3,2,1), fig_dist.add_subplot(3,2,2), fig_dist.add_subplot(3,2,3)
    ax4, ax5, ax6 = fig_dist.add_subplot(3,2,4), fig_dist.add_subplot(3,2,5), fig_dist.add_subplot(3,2,6)
    n_bins = 25
    n16, n50, n68 = np.nanpercentile(HST_NIRCam_dark['redshift'],16),np.nanpercentile(HST_NIRCam_dark['redshift'],50),np.nanpercentile(HST_NIRCam_dark['redshift'],68)
    m16, m50, m68 = np.nanpercentile(MIRIonly_potential['redshift'],16),np.nanpercentile(MIRIonly_potential['redshift'],50),np.nanpercentile(MIRIonly_potential['redshift'],68)
    ax1.hist(HST_NIRCam_dark['redshift'], bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'HST NIRCam dark z=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
    ax1.hist(MIRIonly_potential['redshift'] ,  bins=n_bins ,density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'MIRI onlyz=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68)) 
    ax1.set_xlabel('redshift')
    ax1.legend( prop={'size': 5})

    n16, n50, n68 = np.nanpercentile(HST_NIRCam_dark['mStar'],16),np.nanpercentile(HST_NIRCam_dark['mStar'],50),np.nanpercentile(HST_NIRCam_dark['mStar'],68)
    m16, m50, m68 = np.nanpercentile(MIRIonly_potential['mStar'],16),np.nanpercentile(MIRIonly_potential['mStar'],50),np.nanpercentile(MIRIonly_potential['mStar'],68)
    ax2.hist(HST_NIRCam_dark['mStar'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'log$M_\star$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
    ax2.hist(MIRIonly_potential['mStar'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'log$M_\star$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
    ax2.set_xlabel('log mass')
    ax2.legend( prop={'size': 5})

    n16, n50, n68 = np.nanpercentile(HST_NIRCam_dark['beta'],16),np.nanpercentile(HST_NIRCam_dark['beta'],50),np.nanpercentile(HST_NIRCam_dark['beta'],68)
    m16, m50, m68 = np.nanpercentile(MIRIonly_potential['beta'],16),np.nanpercentile(MIRIonly_potential['beta'],50),np.nanpercentile(MIRIonly_potential['beta'],68)
    ax3.hist(HST_NIRCam_dark['beta'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'$\beta$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
    ax3.hist(MIRIonly_potential['beta'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'$\beta$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
    ax3.set_xlabel('beta')
    ax3.legend( prop={'size': 5})

    n16, n50, n68 = np.nanpercentile(HST_NIRCam_dark['SFR_10'],16),np.nanpercentile(HST_NIRCam_dark['SFR_10'],50),np.nanpercentile(HST_NIRCam_dark['SFR_10'],68)
    m16, m50, m68 = np.nanpercentile(MIRIonly_potential['SFR_10'],16),np.nanpercentile(MIRIonly_potential['SFR_10'],50),np.nanpercentile(MIRIonly_potential['SFR_10'],68)
    ax4.hist(HST_NIRCam_dark['SFR_10'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7  , label=r'SFR$_{10}$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
    ax4.hist(MIRIonly_potential['SFR_10'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'SFR$_{10}$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
    ax4.set_xlabel('SFR_10')
    ax4.legend( prop={'size': 5})

    n16, n50, n68 = np.nanpercentile(HST_NIRCam_dark['SFR_100'],16),np.nanpercentile(HST_NIRCam_dark['SFR_100'],50),np.nanpercentile(HST_NIRCam_dark['SFR_100'],68)
    m16, m50, m68 = np.nanpercentile(MIRIonly_potential['SFR_100'],16),np.nanpercentile(MIRIonly_potential['SFR_100'],50),np.nanpercentile(MIRIonly_potential['SFR_100'],68)
    ax5.hist(HST_NIRCam_dark['SFR_100'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'SFR$_{100}$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
    ax5.hist(MIRIonly_potential['SFR_100'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7, label=r'SFR$_{100}$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
    ax5.set_xlabel('SFR_100')
    ax5.legend( prop={'size': 5})

    n16, n50, n68 = np.nanpercentile(HST_NIRCam_dark['tau'],16),np.nanpercentile(HST_NIRCam_dark['tau'],50),np.nanpercentile(HST_NIRCam_dark['tau'],68)
    m16, m50, m68 = np.nanpercentile(MIRIonly_potential['tau'],16),np.nanpercentile(MIRIonly_potential['tau'],50),np.nanpercentile(MIRIonly_potential['tau'],68)
    ax6.hist(HST_NIRCam_dark['tau'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7, label=r'$\tau$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
    ax6.hist(MIRIonly_potential['tau'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7, label=r'$\tau$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
    ax6.set_xlabel('tau')
    ax6.legend( prop={'size': 5})

    fig_dist.subplots_adjust(wspace=0.2, hspace=0.5)
    if show_plots == True:
        plt.show()
    else:
        None

    return HST_NIRCam_dark , MIRIonly_potential, fig, fig_dist


##################################################################################
#### EVERYTHING FOR THE QUIESCENT GALAXIES
##################################################################################

import os

### now save it in new folder
#current_path = os.getcwd() ## current path
#save_path = current_path + '/JAGUAR_MIRIonly_properties/'
#save_path = '/Users/irisj/Documents/MIRIonly/JAGUAR' + '/JAGUAR_MIRIonly_properties/'
# save_path = '/Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_MIRIonly_properties/' # update

save_path = '/Volumes/champignon/JAGUAR/Outputs/'

# Check if the directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)# If it doesn't exist, create it
    print(f"Directory '{save_path}' created to save the outputs.")
else:
    print(f"Directory '{save_path}' already exists to save the outputs.")


##################################################################################
#### LOAD THE DATA
##################################################################################

path_TC = '/Users/irisj/software/bandpass_plots/'
####### load all the the transmission curve
F435W_TC = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F435W.dat')[:,1]
F606W_TC = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F606W.dat')[:,1]
F775W_TC = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F775W.dat')[:,1]
F814W_TC = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F814W.dat')[:,1]
F850lp_TC = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F850LP.dat')[:,1]

F105W_TC = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F105W.dat')[:,1]
F125W_TC = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F125W.dat')[:,1]
F140W_TC = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F140W.dat')[:,1]
F160W_TC = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F160W.dat')[:,1]

F090W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F090W.dat')[:,1]
F115W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F115W.dat')[:,1]
F150W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F150W.dat')[:,1]
F182M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F182M.dat')[:,1]
F200W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F200W.dat')[:,1]
F210M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F210M.dat')[:,1]
F277W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F277W.dat')[:,1]
F335M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F335M.dat')[:,1]
F356W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F356W.dat')[:,1]
F410M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F410M.dat')[:,1]
F430M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F430M.dat')[:,1]
F444W_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F444W.dat')[:,1]
F460M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F460M.dat')[:,1]
F480M_TC = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F480M.dat')[:,1]

F560W_TC = np.loadtxt(path_TC+'JWST/MIRI/JWST_MIRI.F560W.dat')[:,1]

####### and the corresponding wavelength array
F435W_TC_wl = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F435W.dat')[:,0]
F606W_TC_wl = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F606W.dat')[:,0]
F775W_TC_wl = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F775W.dat')[:,0]
F814W_TC_wl = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F814W.dat')[:,0]
F850lp_TC_wl = np.loadtxt(path_TC+'HST/ACS_WFC/HST_ACS_WFC.F850LP.dat')[:,0]

F105W_TC_wl = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F105W.dat')[:,0]
F125W_TC_wl = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F125W.dat')[:,0]
F140W_TC_wl = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F140W.dat')[:,0]
F160W_TC_wl = np.loadtxt(path_TC+'HST/WFC3_IR/HST_WFC3_IR.F160W.dat')[:,0]

F090W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F090W.dat')[:,0]
F115W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F115W.dat')[:,0]
F150W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F150W.dat')[:,0]
F182M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F182M.dat')[:,0]
F200W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F200W.dat')[:,0]
F210M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F210M.dat')[:,0]
F277W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F277W.dat')[:,0]
F335M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F335M.dat')[:,0]
F356W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F356W.dat')[:,0]
F410M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F410M.dat')[:,0]
F430M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F430M.dat')[:,0]
F444W_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F444W.dat')[:,0]
F460M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F460M.dat')[:,0]
F480M_TC_wl = np.loadtxt(path_TC+'JWST/NIRCam/JWST_NIRCam.F480M.dat')[:,0]

F560W_TC_wl = np.loadtxt(path_TC+'JWST/MIRI/JWST_MIRI.F560W.dat')[:,0]

df_all = pd.DataFrame()
df_all['TC'] = [F435W_TC, F606W_TC, F775W_TC, F814W_TC, F850lp_TC,
                  F105W_TC, F125W_TC, F140W_TC, F160W_TC,
                  F090W_TC, F115W_TC, F150W_TC, F182M_TC, F200W_TC, F210M_TC, F277W_TC,
                  F335M_TC, F356W_TC, F410M_TC, F430M_TC, F444W_TC, F460M_TC, F480M_TC,
                  F560W_TC
                  ]


df_all['TC_wl'] = [F435W_TC_wl, F606W_TC_wl, F775W_TC_wl, F814W_TC_wl, F850lp_TC_wl,
                  F105W_TC_wl, F125W_TC_wl, F140W_TC_wl, F160W_TC_wl,
                  F090W_TC_wl, F115W_TC_wl, F150W_TC_wl, F182M_TC_wl, F200W_TC_wl, F210M_TC_wl, F277W_TC_wl,
                  F335M_TC_wl, F356W_TC_wl, F410M_TC_wl, F430M_TC_wl, F444W_TC_wl, F460M_TC_wl, F480M_TC_wl,
                  F560W_TC_wl
                  ]



###################
## update: open this csv which contains the 1 sigma depth in uJy
###################

# df_depth = pd.read_pickle('/Users/irisj/Documents/GitProjects/depth_code/depth_10nJy_in_midis.pkl').iloc[:-1] # remove the last line which is gabes' midis

# df_depth = df_depth[['band','diam3']]
# print(df_depth)
# ## convert these 1 sigma depth that are in 10nJy to 2.4 time in nJy
# df_depth_10nJy = df_depth['diam3']
# df_depth_nJy = df_depth_10nJy*10

# df_all['2.5depth_nJy'] = 2.5*df_depth_nJy
# print(df_depth_nJy)
# print(df_all['2.5depth_nJy'] )

#### make this new: like take the values from the paper or like table i made fromm the paper

df_depth = pd.read_csv('/Users/irisj/Documents/IrisPhDgithub/Projet1_épuré_v2/df_hist_flux_all_depths_incl_HST.csv')
print(df_depth)
print('Remove the F1000W, F770W, and F560W except v1.4 on 0.040".')
df_depth = df_depth[~df_depth["BAND"].isin(['f1000w deep','f1000w','f770w','midis_60mas_v2','midis_60mas_v1','midis_v2','nc' ])]
print(df_depth)
df_depth = df_depth.iloc[::-1].reset_index(drop=True)
print(df_depth)
## now add 2.5 sigma as i will use that selection as a first, maybe then with the color argument
df_depth['2.5depth_nJy'] = 2.5*df_depth['sig_array_corr']
df_depth['1depth_nJy'] = df_depth['sig_array_corr']
df_depth['5depth_nJy'] = 5*df_depth['sig_array_corr']
print(df_depth)


df_all['LIST'] = [ 'HST_F435W_fnu', 'HST_F606W_fnu', 'HST_F775W_fnu', 'HST_F814W_fnu', 'HST_F850LP_fnu',
                   'HST_F105W_fnu', 'HST_F125W_fnu', 'HST_F140W_fnu', 'HST_F160W_fnu',
                   'NRC_F090W_fnu','NRC_F115W_fnu','NRC_F150W_fnu','NRC_F182M_fnu','NRC_F200W_fnu','NRC_F210M_fnu','NRC_F277W_fnu',
                   'NRC_F335M_fnu','NRC_F356W_fnu','NRC_F410M_fnu','NRC_F430M_fnu','NRC_F444W_fnu','NRC_F460M_fnu','NRC_F480M_fnu',
                   'F560W' ]
df_all['5depth_nJy'] = df_depth['5depth_nJy'] 
df_all['2.5depth_nJy'] = df_depth['2.5depth_nJy'] 
df_all['1depth_nJy'] = df_depth['1depth_nJy'] 
df_all['BAND'] = df_depth['BAND']
##################################################################################
#### 
##################################################################################

#R = 'r1'

for R in ['r4']: # 
# for R in ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10']:
 
    SF_spectra_part1 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_0p2_1.fits'%(R,R)
    SF_spectra_part2 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_1_1p5.fits'%(R,R)
    SF_spectra_part3 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_1p5_2.fits'%(R,R)
    SF_spectra_part4 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_2_3.fits'%(R,R)
    SF_spectra_part5 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_3_4.fits'%(R,R)
    SF_spectra_part6 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_4_5.fits'%(R,R)
    SF_spectra_part7 =  '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_5_15.fits'%(R,R)

    ## selects the HST NIRCam and MIRI only potential
    Q_SF_combined_MIRI = Table.read('/Volumes/champignon/JAGUAR/Outputs/%s/JAGUAR_%s_MIRI.fits'%(R,R))
    HST_NIRCam_dark , MIRIonly_potential, fig, fig_dist = FindMIRIonlyAndPlot(Q_SF_combined_MIRI,df_all,R)
    fig.savefig('/Volumes/champignon/JAGUAR/Outputs/%s/JAGUAR_properties_%s_fig_color_selection.pdf'%(R,R))
    fig_dist.savefig('/Volumes/champignon/JAGUAR/Outputs/%s/JAGUAR_properties_%s_fig_dist_color_selection.pdf'%(R,R))
    # plt.show()
    # exit()

    ### uncomment that line if want to save if without the wl
    MIRIonly_potential.write(save_path+'%s/MIRIonly_potential_%s_color_selection.fits'%(R,R),overwrite=True)
    # exit()

#'''  
    ## extracts the corresponding SEDs to these galaxies and makes the pretty plots
    # stores them in that df_spectra
    colnames = ['ID','redshift','TYPE','WL', 'SED']
    df_spectra = pd.DataFrame()


    ''' 
    If the boolean to extract the spectra is true, it will extract the spectra from here
    Otherwise it will put np.zeros of some length in that position
    '''
    # here the loop starts, loops over all the potential miri potential i.e. the ones
    # with HST and NIRCam filters below the 2.5 depth and above 2.5 depth in MIRI
    # it first checks if it is a QG or a SFG
    if extract_spectra == True:
        print('JAGUAR Spectra extraction for realization %s started. This might take a bit of time if there are many spectra to extract.'%(R))
        for i in range(len(MIRIonly_potential)): ## loop over all the MIRI potentials
            this_type = MIRIonly_potential['TYPE'][i]
            this_id = MIRIonly_potential['ID'][i]
            this_z = MIRIonly_potential['redshift'][i]
            print('This MIRI only is a %s galaxy, with z=%.3f and ID=%i.'%(this_type,this_z,this_id))

            #print('DEBUGGING EXIT')
            #exit()        
            if this_type == 'QG':
                # 1) open the data, need the id to match the id
                path_QG_spectra = '/Volumes/champignon/JAGUAR/%s/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits'%(R,R)
                SPECTRA_tab =  np.array( fits.open(path_QG_spectra)[1].data)
                WL_tab = np.array([fits.open(path_QG_spectra)[2].data] * len(SPECTRA_tab))
                id_tab = fits.open(path_QG_spectra)[3].data['ID']
                # 2) save the data with matching id 
                ix = np.where( id_tab == this_id )[0][0]
                the_SPECTRA = SPECTRA_tab[ix]
                the_WL = WL_tab[ix]
                print('ix',ix)


            elif this_type == 'SFR':
                ## if it's a SFGs need to check the redshift
                this_z = MIRIonly_potential['redshift'][i]
                if (this_z > 0.2) & (this_z <= 1):
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part1
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)

                elif (this_z > 1) & (this_z <= 1.5):
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part2
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)

                elif (this_z > 1.5) & (this_z <= 2.0):
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part3
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)

                elif (this_z > 2.0) & (this_z <= 3.0):
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part4
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)

                elif (this_z > 3.0) & (this_z <= 4.0):
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part5
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)

                elif (this_z > 4.0) & (this_z <= 5.0):
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part6
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)

                elif (this_z > 5.0) & (this_z <= 15.0):
                    print('CHECK')
                    # 1) open the data, need the id to match the id
                    SF_spectra_name = SF_spectra_part7
                    SPECTRA_tab = np.array( fits.open(SF_spectra_name)[1].data)
                    WL_tab = np.array([fits.open(SF_spectra_name)[2].data] * len(SPECTRA_tab))
                    id_tab = fits.open(SF_spectra_name)[3].data['ID']
                    # 2) save the data with matching id 
                    ix = np.where( id_tab == this_id )[0][0]
                    the_SPECTRA = SPECTRA_tab[ix]
                    the_WL = WL_tab[ix]
                    print('ix',ix)
                    #print('SPECTRA', the_SPECTRA)
                    #print( 'WL', the_WL)

                else:
                    print('Something is wrong. Check the code. <3')
                    exit()


                data = [str(this_id),float(this_z),str(this_type),np.array(the_WL,dtype=np.float32),np.array(the_SPECTRA,dtype=np.float32)]
                print( 'data : \n', data)
                # colnames = ['ID','redshift','TYPE','WL', 'SED']
                new_row = pd.Series(data,index=colnames) ## maybe try to specify the data type here??
                print('new row : \n', new_row)
                df_spectra = pd.concat([df_spectra, new_row.to_frame().T], ignore_index=True)
                ## create a table and then select the line and make it append to the df_spectra

            else:
                print('Something is wrong. Check the code. <3')
                exit()

    else: # extract_spectra == False:
        print('JAGUAR Spectra extraction for realization %s is not done. The table with store an array of length 99 containing only 0 in there.'%(R))
        ## just take the length of one of the spectra
        SF_spectra_part1 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_0p2_1.fits'%(R,R)
        path_QG_spectra = '/Volumes/champignon/JAGUAR/%s/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits'%(R,R)
        len_SPECTRA_tab =  len(np.array( fits.open(path_QG_spectra)[1].data))
        print(len_SPECTRA_tab)
        #exit()
        #WL_tab = np.array([fits.open(path_QG_spectra)[2].data] * len(SPECTRA_tab))
        
        the_WL = np.ones(len_SPECTRA_tab)
        the_SPECTRA = np.ones(len_SPECTRA_tab)
        for i in range(len(MIRIonly_potential)): ## loop over all the MIRI potentials
            this_type = MIRIonly_potential['TYPE'][i]
            this_id = MIRIonly_potential['ID'][i]
            this_z = MIRIonly_potential['redshift'][i]
            print('This MIRI only is a %s galaxy, with z=%.3f and ID=%i.'%(this_type,this_z,this_id))
            data = [str(this_id),float(this_z),str(this_type),np.array(the_WL,dtype=np.float32),np.array(the_SPECTRA,dtype=np.float32)]
            print( 'data : \n', data)
            # colnames = ['ID','redshift','TYPE','WL', 'SED']
            new_row = pd.Series(data,index=colnames) ## maybe try to specify the data type here??
            print('new row : \n', new_row)
            df_spectra = pd.concat([df_spectra, new_row.to_frame().T], ignore_index=True)
        
    print(df_spectra)
    # csv saves all as strings and astropy table fails
    df_spectra.to_pickle(save_path+'%s/MIRIonly_%s_color_selection.pkl'%(R,R))

    # here saves the photometry and the spectra etc there
    # MIRIonly_potential['WL'] = df_spectra['WL'].tolist()
    # MIRIonly_potential['SED'] = df_spectra['SED'].tolist()
    # #print(MIRIonly_potential)
    
    # MIRIonly_potential.write(save_path+'MIRIonly_potential_%s.fits'%(R),overwrite=True)
    # print(save_path+'MIRIonly_potential_%s.fits'%(R))

    print('Realisation %s is done.\n'%R)
