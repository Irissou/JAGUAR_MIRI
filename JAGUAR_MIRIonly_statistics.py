'''
Iris Jermann - PhD Student DAWN/DTU

This code: From JAGUAR simulation understand what are the properties of MIRI only.

Print and study the average properties of all the catalogues

Update october 2024: remove all the not miri only potential and focus only on the MIRI

Make the final figure like on my Brouillon :)) and only use the spectra from R1 

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
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
#import pyphot
#from JAGUAR_OpenData import *
import sys
sys.path.append('/Users/irisj/Documents/MIRIonly/everything_on_v7_pablo')
from OpenData_v7_2024 import *



###################
## update: open this csv which contains the 1 sigma depth in uJy
###################

df_depth = pd.read_pickle('/Users/irisj/Documents/GitProjects/depth_code/depth_10nJy_in_midis.pkl').iloc[:-1] # remove the last line which is gabes' midis

df_depth = df_depth[['band','diam3']]
print(df_depth)
## convert these 1 sigma depth that are in 10nJy to 2.4 time in nJy
df_depth_10nJy = df_depth['diam3']
df_depth_nJy = df_depth_10nJy*10

df_all['2.5depth nJy'] = 2.5*df_depth_nJy
print(df_depth_nJy)
print(df_all['2.5depth nJy'] )

##################################################################################
#### How I define the 3 groups:
# RED: ALL z < 6.8
# GREEN: ALL 6.8 <= z <= 8.3 
# BLUE: ALL 9.5 <= z <= 11.2

##################################################################################




##################################################################################
#### 
##################################################################################

import os


#current_path = os.getcwd() ## current path
save_path = '/Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_MIRIonly_statistics/'
# Check if the directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)# If it doesn't exist, create it
    print(f"Directory '{save_path}' created to save the outputs.")
else:
    print(f"Directory '{save_path}' already exists to save the outputs.")



# save_path = '/Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_MIRIonly_statistics_depth'

show_plots = False

##################################################################################
#### Import the catalogs for the 10 realisations containing the spectra

####### for all
all_r = []

#for R in ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10']: 

    #all_r.append(Table.read('/Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_MIRIonly_properties/MIRIonly_potential_%s.fits'%R))

## R1 I extracted the spectra
R = 'r1'
this_tab = Table.read('/Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_MIRIonly_properties/MIRIonly_potential_%s.fits'%R)
this_tab['R'] = R ## add the R string to each gal, so can do stat later
all_r.append(this_tab)

## R2 to R10 I didn't extract the spectra bc at this point point less, so I put arrays with 0 in these columns, so the code can still run

for R in ['r2','r3','r4','r5','r6','r7','r8','r9','r10']: 
    this_tab = Table.read('/Users/irisj/Documents/MIRIonly/JAGUAR/JAGUAR_MIRIonly_properties/MIRIonly_potential_%s.fits'%R)
    this_tab['R'] = R
    all_r.append(this_tab)

fits_r = vstack(all_r)


print('MIRI only sources table:')
print(fits_r)


##################################################################################
#### For each make a plot with the median one and the 19, 25 in very fine 
# overplotting all the limits ! including mips and spitzer ALMA blabla
##################################################################################

plt.rcParams['font.size'] = 12
cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Om0=0.308)

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(212)
ax_z, ax_logM, ax_beta = fig.add_subplot(4,3,1),fig.add_subplot(4,3,2),fig.add_subplot(4,3,3)
ax_sfr10, ax_sfr100, ax_tau = fig.add_subplot(4,3,4),fig.add_subplot(4,3,5),fig.add_subplot(4,3,6)
#fig.suptitle('Spectral Energy Distribution (SED) of a Single Stellar Population (SSP)\nAge %.1f Myr and M$\star$ %.3e'%(the_age,the_mstar),size=16)
#first_legend.get_frame().set_facecolor('white')

fig.subplots_adjust(wspace=0, hspace=0)

##################################################################################
### top pannels with the distribution of the properties
##################################################################################

ax_z.tick_params( labelbottom=False, labelleft=True, labeltop=True, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_logM.tick_params( labelbottom=False, labelleft=False, labeltop=True, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_beta.tick_params( labelbottom=False, labelleft=False, labeltop=True, labelright=True, direction='inout',bottom=True, top=True, left=True, right=True)
ax_sfr10.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_sfr100.tick_params( labelbottom=True, labelleft=False, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_tau.tick_params( labelbottom=True, labelleft=False, labeltop=False, labelright=True, direction='inout',bottom=True, top=True, left=True, right=True)

ax_z.xaxis.set_label_position('top')
ax_logM.xaxis.set_label_position('top')
ax_beta.xaxis.set_label_position('top')

### add titles but inside (HST_NIRCam_dark photometric properties are in fits_r_...)
ax_z.text(0.5, 0.95, r'$z$', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))
n_bins = np.arange(0,15.1,0.5)

if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['redshift'],16),np.nanpercentile(fits_r['redshift'],50),np.nanpercentile(fits_r['redshift'],84)
    ax_z.hist(fits_r['redshift'] ,  bins=n_bins ,density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'MIRI onlyz=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68)) 
ax_z.set_xlabel('redshift')
ax_z.legend( prop={'size': 5})
ax_z.axvline(8.3, color='gray', ls='--')
ax_z.axvline(9.1, color='gray', ls='--' )
ax_z.axvline(11.55, color='gray', ls='--')



n_bins = np.arange(5.5,8.6,0.25)
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['mStar'],16),np.nanpercentile(fits_r['mStar'],50),np.nanpercentile(fits_r['mStar'],84)
    ax_logM.hist(fits_r['mStar'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'log$M_\star$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_logM.set_xlabel('log mass')
ax_logM.legend( prop={'size': 5})


n_bins = np.arange(-3.,-0.4,0.25)
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['beta'],16),np.nanpercentile(fits_r['beta'],50),np.nanpercentile(fits_r['beta'],84)
    ax_beta.hist(fits_r['beta'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'$\beta$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_beta.set_xlabel('beta')
ax_beta.legend( prop={'size': 5})

n_bins = np.arange(-4.,1.1,0.3)
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['SFR_10'],16),np.nanpercentile(fits_r['SFR_10'],50),np.nanpercentile(fits_r['SFR_10'],84)
    ax_sfr10.hist(fits_r['SFR_10'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'SFR$_{10}$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_sfr10.set_xlabel('SFR_10')
ax_sfr10.legend( prop={'size': 5})


n_bins = np.arange(-4.,1.1,0.3)
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['SFR_100'],16),np.nanpercentile(fits_r['SFR_100'],50),np.nanpercentile(fits_r['SFR_100'],84)
    ax_sfr100.hist(fits_r['SFR_100'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7, label=r'SFR$_{100}$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_sfr100.set_xlabel('SFR_100')
ax_sfr100.legend( prop={'size': 5})


n_bins = np.arange(6.5,11.6,0.25)
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['tau'],16),np.nanpercentile(fits_r['tau'],50),np.nanpercentile(fits_r['tau'],84)
    ax_tau.hist(fits_r['tau'] , bins= n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7, label=r'$\tau$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_tau.set_xlabel('tau')
ax_tau.legend( prop={'size': 5})

plt.show()
exit()

##################################################################################
### group the families in bins of redshift
##################################################################################

########### MIRI only
MIRIonly_g1 = fits_r[fits_r['redshift']<8.29] ## limit for being Ha excess
MIRIonly_g2 = fits_r[(fits_r['redshift']>=9.1)&(fits_r['redshift']<11.55)] ## begin Hb oiii exceess, 
MIRIonly_g3 = fits_r[fits_r['redshift']>=11.55] ## combination of balmer break and OIII Hb


print('Number of miri sources in group 1: ', len(MIRIonly_g1))
print(np.average(MIRIonly_g1['redshift']),np.average(MIRIonly_g1['mStar']),np.average(MIRIonly_g1['beta']),
      np.average(MIRIonly_g1['SFR_10']),np.average(MIRIonly_g1['SFR_100']),np.average(MIRIonly_g1['tau']))
print('Number of miri sources in group 2: ', len(MIRIonly_g2))
print(np.average(MIRIonly_g2['redshift']),np.average(MIRIonly_g2['mStar']),np.average(MIRIonly_g2['beta']),
      np.average(MIRIonly_g2['SFR_10']),np.average(MIRIonly_g2['SFR_100']),np.average(MIRIonly_g2['tau']))
print('Number of miri sources in group 3: ', len(MIRIonly_g3))
print(np.average(MIRIonly_g3['redshift']),np.average(MIRIonly_g3['mStar']),np.average(MIRIonly_g3['beta']),
      np.average(MIRIonly_g3['SFR_10']),np.average(MIRIonly_g3['SFR_100']),np.average(MIRIonly_g3['tau']))



##################################################################################
### bottom subplot with the average SED
##################################################################################
all_A = []
all_B = []
for i in range(len(df_all.iloc[:25])):
    A, = ax.plot(df_all['TC_wl'][i]/10000,df_all['TC'][i]/1000,color=df_all['COLOR'][i],label=df_all['INSTR'][i]) #,label=df['band'][i])
    B = ax.fill_between(df_all['TC_wl'][i]/10000,df_all['TC'][i]/1000, color=df_all['COLOR'][i], alpha=0.5,label=df_all['BAND'][i])
    all_A.append(A)
    all_B.append(B)
    ##### ADD HERE THE LIMITS I USED TO SELECT THEM
    ax.errorbar(df_all['wl'][i], df_all['2.5depth nJy'][i],fmt='o', capsize=5, color='gray' ) #, color=df_all['COLOR'][i])
    #df_sel['2.5depth nJy'][:25].values.tolist() ## also plot the

#ax.set_xlim(0,12) ## on veut voir les autres MIRI bands !!
ax.set_ylim(1e-6,1e4)
ax.set_xlim(0.25e0,1.5e1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\lambda_{obs}$ [$\mu$m]',size=20)
#ax.set_ylabel(r'F$_\lambda^{obs}$/1e-26 [erg s$^{-1}$ Å$^{-1}$]')
ax.set_ylabel(r'f$_\nu^{obs}$ [nJy]',size=20)

#second_legend = ax.legend( handles = [all_A[0],all_A[5],all_A[9],all_A[11],all_A[-1]],prop={'size': 12},frameon= True,  framealpha=0.7,loc='upper left', bbox_to_anchor=(1, 1)) #loc=2,
#ax.add_artist(second_legend)

#third_legend = ax.legend( handles = all_B,prop={'size': 12},loc=1,frameon= True, framealpha=0.7,bbox_to_anchor=(1.13, 1))
#ax.add_artist(third_legend)

#### add the mag on the right
'''
ax_mag = ax.twinx()
ax_mag.set_ylabel('mag')
ax_mag.set_ylim(-2.5*np.log10(1e-6 * 1e-6 ) + 8.9 , -2.5*np.log10(1e1 * 1e-6) + 8.9 ) ## need to change that for the second y axis to match with the limits

for i in range(len(df_all)):
    centi = df_all['wl'][i]
    dd = df_all['2.5depth_muJy'][i]
    couleur = df_all['COLOR'][i]
    ax_mag.plot( [ centi-0.15,centi+0.15] , [ dd, dd] ,  lw = 2.5 , c = couleur ) ## on y the depth , and on x the center pm 0.15 mum

#first_legend = ax.legend( handles = [P2,PD],prop={'size': 12},loc=1,frameon= True, framealpha=0.7)
#ax.add_artist(first_legend)
ax.tick_params(axis='both',which='major',direction='inout',width=1,length=7,bottom=True, top=True,labelbottom=True,labeltop=False)
ax.tick_params(axis='both',which='minor',direction='inout',width=1,length=3,bottom=True, top=True,labelbottom=True,labeltop=False)
#ax[2].tick_params(axis='y',which='both',width=3,labelleft=True,labelright=True, labelsize=20) 

fig.subplots_adjust(left=0.08,right=0.88,top=0.88,bottom=0.12)
'''
############### now add the science i.e. the SED - the median the ~20 individual ones - of the MIRI ONLY 

all_wl_obs_MIRIonly = []
all_s_obs_MIRIonly = []
for i in range( len(fits_r)): # looping on each galaxy in the table 

    ## since we want in OBSERVED frame -> divide the flux by (1+z) and multiply wavelength by (1+z)
    z = fits_r['redshift'][i]

    ### the x axis: wavelength array
    wl_obs = fits_r['WL'][i]*(1+z) ## since i don't give a unit in here i have to give it in u.spectral_density

    ### the y axis: flux array -> convert from u.erg/u.s/u.cm**2/u.AA to nJy and then from RF to OBS frame
    s_cgs_obs = (fits_r['SED'][i]/(1+z)) * u.erg/u.s/u.cm**2/u.AA
    ## convert into nanoJy
    s_nJy_obs = s_cgs_obs.to(u.nJy, u.spectral_density(u.AA, wl_obs) ).value

    ### the x axis: wavelength array
    wl_obs = (wl_obs * u.AA ).to('micron').value # convert it into microns
    # to take then the median 
    all_wl_obs_MIRIonly.append(wl_obs)
    all_s_obs_MIRIonly.append(s_nJy_obs)
    ''' 
    ## ça revient au même de faire le redshift avant ou après mettre en muJ<
    #wl = fits_r['WL'][i]*(1+z) ## since i don't give a unit in here i have to give it in u.spectral_density
    #s_cgs = (fits_r['SED'][i]/(1+z)) * u.erg/u.s/u.cm**2/u.AA
    #s_muJy = s_cgs.to(u.mJy, u.spectral_density(u.AA, wl) ).value
    #print(s_muJy
    #fig_test = plt.figure()
    #ax_test = fig_test.add_subplot(111)
    #ax_test.plot(s_muJy,s_muJy_obs)
    #plt.show()
    '''
    ## plot individual spectra
    #ax.plot(wl_obs, s_nJy_obs, color='black',lw='1')

all_wl_obs_MIRIonly = np.array(all_wl_obs_MIRIonly)
all_s_obs_MIRIonly = np.array(all_s_obs_MIRIonly)




#### now in bins 
# use the length of the groups to define the indices
for l_down, l_up in zip([0,len(MIRIonly_g1),len(MIRIonly_g2)],[len(MIRIonly_g1),len(MIRIonly_g2),len(MIRIonly_g3)]):
    med_wl_obs_MIRIonly = np.nanmedian(all_wl_obs_MIRIonly[l_down:l_up],axis = 0)
    med_s_obs_MIRIonly = np.nanmedian(all_s_obs_MIRIonly[l_down:l_up],axis = 0)
    ax.plot(med_wl_obs_MIRIonly, med_s_obs_MIRIonly, color='crimson',lw='2.5',alpha = 0.7)


#plt.rcParams['image.cmap'] = 'Grays'
#plt.rcParams['figure.figsize'] = (14,10)
#fig.savefig('/Users/irisj/Downloads/SED_goodone.png', dpi=380,format='png')
#fig.savefig('/Users/irisj/Downloads/SED_justSFR.png', dpi=380,format='png')

if show_plots == True:
        plt.show()
else:
    None

fig.savefig('LETTER_MIRIonlyFigure4.png', dpi=380,format='png')


exit()


####################################################################################################################################################################
### SPLIT THE FIGURE INTO 2 : ONE WITH THE HISTOGRAMS AND ONE WITH THE SED BUT ONLY MIRI ONLY
####################################################################################################################################################################


fig_HISTOGRAMS = plt.figure(figsize=(10,5.5))
ax_z, ax_logM, ax_beta = fig_HISTOGRAMS.add_subplot(2,3,1),fig_HISTOGRAMS.add_subplot(2,3,2),fig_HISTOGRAMS.add_subplot(2,3,3)
ax_sfr10, ax_sfr100, ax_tau = fig_HISTOGRAMS.add_subplot(2,3,4),fig_HISTOGRAMS.add_subplot(2,3,5),fig_HISTOGRAMS.add_subplot(2,3,6)

fig_HISTOGRAMS.subplots_adjust(wspace=0.2, hspace=0.4, left=0.1, right=0.9, top=0.8, bottom=0.15)



##################################################################################
### top pannels with the distribution of the properties
##################################################################################

ax_z.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_logM.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_beta.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_sfr10.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_sfr100.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)
ax_tau.tick_params( labelbottom=True, labelleft=True, labeltop=False, labelright=False, direction='inout',bottom=True, top=True, left=True, right=True)

#ax_z.xaxis.set_label_position('top')
#ax_logM.xaxis.set_label_position('top')
#ax_beta.xaxis.set_label_position('top')

### add titles but inside (HST_NIRCam_dark photometric properties are in fits_r_...)
ax_z.text(0.5, 0.95, r'$z$', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))
n_bins = np.arange(0,15.1,0.5)
n16, n50, n68 = np.nanpercentile(fits_r_hstdark['redshift'],16),np.nanpercentile(fits_r_hstdark['redshift'],50),np.nanpercentile(fits_r_hstdark['redshift'],68)
m16, m50, m68 = np.nanpercentile(fits_r_4micron['redshift'],16),np.nanpercentile(fits_r_4micron['redshift'],50),np.nanpercentile(fits_r_4micron['redshift'],68)
#ax_z.hist(fits_r_hstdark['redshift'], bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'z=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68) ) # label=r'HST NIRCam non-detections z=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
#ax_z.hist(fits_r_4micron['redshift'] ,  bins=n_bins ,density=True, color='forestgreen', edgecolor='black', alpha=0.7 ,label=r'z=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68)  ) #label=r'F444W dropouts z=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68)) 
Z1_2 = ax_z.hist(fits_r_hstdark['redshift'], bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label='HST NIRCam non-detections' ) # label=r'HST NIRCam non-detections z=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
Z2_2 = ax_z.hist(fits_r_4micron['redshift'] ,  bins=n_bins ,density=True, color='forestgreen', edgecolor='black', alpha=0.7 ,label='F444W dropouts'  )
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['redshift'],16),np.nanpercentile(fits_r['redshift'],50),np.nanpercentile(fits_r['redshift'],68)
    #_,_,Z3 = ax_z.hist(fits_r['redshift'] ,  bins=n_bins ,density=True, color='crimson', edgecolor='black', alpha=0.7 ,label=r'z=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68) ) #label=r'MIRI only z=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68)) 
    Z3_2 = ax_z.hist(fits_r['redshift'] ,  bins=n_bins ,density=True, color='crimson', edgecolor='black', alpha=0.7 ,label='MIRI only')
ax_z.set_xlabel('redshift')

L1 = ax_z.axvline(6.8, color='gray', ls='dashed')#,label=r'z=6.8 (lower limit H$\alpha$ excess)')
L2 = ax_z.axvline(8.3, color='gray', ls='dashed')#,label=r'z=8.3 (upper limit H$\alpha$ excess)')
L3 = ax_z.axvline(9.2, color='gray', ls='dotted')#, label=r'z=9.2 (lower limit H$\beta$+[OIII] excess)' )
L4 = ax_z.axvline(11.6, color='gray', ls='dotted' )#, label=r'z=11.6 (upper limit H$\beta$+[OIII] excess)' )
ax_z.legend( prop={'size': 5},bbox_to_anchor=(0.1, 1.1))
#first_Z_legend = ax_z.legend(handles = [L1,L2,L3,L4], prop={'size': 5},bbox_to_anchor=(1, 1.5))
#ax_z.add_artist(first_Z_legend)
#second_Z_legend =  ax_z.legend(handles = [Z1], prop={'size': 5})
#ax_z.add_artist(second_Z_legend)
#second_legend = ax_SED.legend( handles = [all_CC[0],all_CC[-1]],prop={'size': 8},frameon= True,  framealpha=0.7, bbox_to_anchor=(1, 1)) #loc=2,
ax_z.set_ylim(0,0.8)

n_bins = np.arange(5.5,8.6,0.25)
n16, n50, n68 = np.nanpercentile(fits_r_hstdark['mStar'],16),np.nanpercentile(fits_r_hstdark['mStar'],50),np.nanpercentile(fits_r_hstdark['mStar'],68)
m16, m50, m68 = np.nanpercentile(fits_r_4micron['mStar'],16),np.nanpercentile(fits_r_4micron['mStar'],50),np.nanpercentile(fits_r_4micron['mStar'],68)
ax_logM.hist(fits_r_hstdark['mStar'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'log$M_\star$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
ax_logM.hist(fits_r_4micron['mStar'] , bins=n_bins , density=True, color='forestgreen', edgecolor='black', alpha=0.7 , label=r'log$M_\star$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['mStar'],16),np.nanpercentile(fits_r['mStar'],50),np.nanpercentile(fits_r['mStar'],68)
    ax_logM.hist(fits_r['mStar'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'log$M_\star$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_logM.set_xlabel('log mass')
ax_logM.legend( prop={'size': 5})


n_bins = np.arange(-3.,-0.4,0.25)
n16, n50, n68 = np.nanpercentile(fits_r_hstdark['beta'],16),np.nanpercentile(fits_r_hstdark['beta'],50),np.nanpercentile(fits_r_hstdark['beta'],68)
m16, m50, m68 = np.nanpercentile(fits_r_4micron['beta'],16),np.nanpercentile(fits_r_4micron['beta'],50),np.nanpercentile(fits_r_4micron['beta'],68)
ax_beta.hist(fits_r_hstdark['beta'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'$\beta$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
ax_beta.hist(fits_r_4micron['beta'] , bins=n_bins , density=True, color='forestgreen', edgecolor='black', alpha=0.7 , label=r'$\beta$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['beta'],16),np.nanpercentile(fits_r['beta'],50),np.nanpercentile(fits_r['beta'],68)
    ax_beta.hist(fits_r['beta'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'$\beta$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_beta.set_xlabel('beta')
ax_beta.legend( prop={'size': 5})

n_bins = np.arange(-4.,1.1,0.3)
n16, n50, n68 = np.nanpercentile(fits_r_hstdark['SFR_10'],16),np.nanpercentile(fits_r_hstdark['SFR_10'],50),np.nanpercentile(fits_r_hstdark['SFR_10'],68)
m16, m50, m68 = np.nanpercentile(fits_r_4micron['SFR_10'],16),np.nanpercentile(fits_r_4micron['SFR_10'],50),np.nanpercentile(fits_r_4micron['SFR_10'],68)
ax_sfr10.hist(fits_r_hstdark['SFR_10'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7  , label=r'SFR$_{10}$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
ax_sfr10.hist(fits_r_4micron['SFR_10'] , bins=n_bins , density=True, color='forestgreen', edgecolor='black', alpha=0.7 , label=r'SFR$_{10}$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['SFR_10'],16),np.nanpercentile(fits_r['SFR_10'],50),np.nanpercentile(fits_r['SFR_10'],68)
    ax_sfr10.hist(fits_r['SFR_10'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7 , label=r'SFR$_{10}$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_sfr10.set_xlabel('SFR_10')
ax_sfr10.legend( prop={'size': 5})


n_bins = np.arange(-4.,1.1,0.3)
n16, n50, n68 = np.nanpercentile(fits_r_hstdark['SFR_100'],16),np.nanpercentile(fits_r_hstdark['SFR_100'],50),np.nanpercentile(fits_r_hstdark['SFR_100'],68)
m16, m50, m68 = np.nanpercentile(fits_r_4micron['SFR_100'],16),np.nanpercentile(fits_r_4micron['SFR_100'],50),np.nanpercentile(fits_r_4micron['SFR_100'],68)
ax_sfr100.hist(fits_r_hstdark['SFR_100'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7 , label=r'SFR$_{100}$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
ax_sfr100.hist(fits_r_4micron['SFR_100'] , bins=n_bins , density=True, color='forestgreen', edgecolor='black', alpha=0.7, label=r'SFR$_{100}$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['SFR_100'],16),np.nanpercentile(fits_r['SFR_100'],50),np.nanpercentile(fits_r['SFR_100'],68)
    ax_sfr100.hist(fits_r['SFR_100'] , bins=n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7, label=r'SFR$_{100}$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_sfr100.set_xlabel('SFR_100')
ax_sfr100.legend( prop={'size': 5})


n_bins = np.arange(6.5,11.6,0.25)
n16, n50, n68 = np.nanpercentile(fits_r_hstdark['tau'],16),np.nanpercentile(fits_r_hstdark['tau'],50),np.nanpercentile(fits_r_hstdark['tau'],68)
m16, m50, m68 = np.nanpercentile(fits_r_4micron['tau'],16),np.nanpercentile(fits_r_4micron['tau'],50),np.nanpercentile(fits_r_4micron['tau'],68)
ax_tau.hist(fits_r_hstdark['tau'] ,  bins=n_bins , density=True, color='skyblue', edgecolor='black', alpha=0.7, label=r'$\tau$=%.3f$^{%.3f}_{%.3f}$'%(n16,n50,n68))
ax_tau.hist(fits_r_4micron['tau'] , bins=n_bins , density=True, color='forestgreen', edgecolor='black', alpha=0.7, label=r'$\tau$=%.3f$^{%.3f}_{%.3f}$'%(m16,m50,m68))
if len(fits_r) == 0:
    None
else:
    l16, l50, l68 = np.nanpercentile(fits_r['tau'],16),np.nanpercentile(fits_r['tau'],50),np.nanpercentile(fits_r['tau'],68)
    ax_tau.hist(fits_r['tau'] , bins= n_bins , density=True, color='crimson', edgecolor='black', alpha=0.7, label=r'$\tau$=%.3f$^{%.3f}_{%.3f}$'%(l16,l50,l68))
ax_tau.set_xlabel('tau')
ax_tau.legend( prop={'size': 6})


#### legend mess for the histrograms: on top create a legend with the colors and the categorie
# then in the plots put the median values
# on top too place the redshift limits for Ha and OIII Hb emitters



fig_HISTOGRAMS.savefig('LETTER_MIRIonlyFigure4_HISTOGRAMS.png', dpi=380,format='png')



##################################################################################
### bottom pannel with the SEDs - only the MIRI ones
##################################################################################

fig_SED = plt.figure(figsize=(9,4))
ax_SED = fig_SED.add_subplot(111)

########### MIRI only
print('JUST DOUBLE CHECK: ',fits_r['redshift'])
MIRIonly_g1 = fits_r[fits_r['redshift']<8.3] ## limit for being Ha excess
MIRIonly_g2 = fits_r[(fits_r['redshift']>=9.2)&(fits_r['redshift']<11.6)] ## begin Hb oiii exceess, 
MIRIonly_g3 = fits_r[fits_r['redshift']>=11.6] ## combination of balmer break and OIII Hb

print('MIRI ONLY G1')
print(MIRIonly_g1)

print('\n\nMIRI ONLY G2')
print(MIRIonly_g2)

print('\n\nMIRI ONLY G3')
print(MIRIonly_g3)
print(np.percentile(MIRIonly_g3['redshift'],50), np.percentile(MIRIonly_g3['redshift'],50)-np.percentile(MIRIonly_g3['redshift'],16),np.percentile(MIRIonly_g3['redshift'],84)-np.percentile(MIRIonly_g3['redshift'],50))
print(np.percentile(MIRIonly_g3['mStar'],50), np.percentile(MIRIonly_g3['mStar'],50)-np.percentile(MIRIonly_g3['mStar'],16),np.percentile(MIRIonly_g3['mStar'],84)-np.percentile(MIRIonly_g3['redshift'],50))

MIRIonly_g1_wl = np.nanmedian(all_wl_obs_MIRIonly[:len(MIRIonly_g1)],axis = 0)
MIRIonly_g1_s = np.nanmedian(all_s_obs_MIRIonly[:len(MIRIonly_g1)],axis = 0)
MIRIonly_g2_wl = np.nanmedian(all_wl_obs_MIRIonly[len(MIRIonly_g1):len(MIRIonly_g2)],axis = 0)
MIRIonly_g2_s = np.nanmedian(all_s_obs_MIRIonly[len(MIRIonly_g1):len(MIRIonly_g2)],axis = 0)
MIRIonly_g3_wl =  np.nanmedian(all_wl_obs_MIRIonly[len(MIRIonly_g2):len(MIRIonly_g3)],axis = 0)
MIRIonly_g3_s = np.nanmedian(all_s_obs_MIRIonly[len(MIRIonly_g2):len(MIRIonly_g3)],axis = 0)


V1, = ax_SED.plot(MIRIonly_g1_wl, MIRIonly_g1_s, color='tomato', ls='dotted', lw='2.5',alpha = 0.9, label=r'H$\alpha$ excess')
V2, = ax_SED.plot(MIRIonly_g2_wl, MIRIonly_g2_s, color='red',lw='2.5',alpha = 0.9, label=r'H$\beta$+[OIII] excess')
V3, = ax_SED.plot(MIRIonly_g3_wl, MIRIonly_g3_s, color='crimson', ls='dashed',lw='2.5',alpha = 0.9, label='z>11.6 solution') # label=r'Balmer break+emission lines?')
first_legend = ax_SED.legend( handles = [V1,V2,V3],prop={'size': 8},frameon= True,  framealpha=0.7, bbox_to_anchor=(0.08, 0.78)) #loc=2,
ax_SED.add_artist(first_legend)
#ax_SED.legend(fontsize = 8, bbox_to_anchor=(0.1, 0.8))

print('Number of miri sources in group 1: ', len(MIRIonly_g1))
print(np.average(MIRIonly_g1['redshift']),np.average(MIRIonly_g1['mStar']),np.average(MIRIonly_g1['beta']),
      np.average(MIRIonly_g1['SFR_10']),np.average(MIRIonly_g1['SFR_100']),np.average(MIRIonly_g1['tau']))
print('Number of miri sources in group 2: ', len(MIRIonly_g2))
print(np.average(MIRIonly_g2['redshift']),np.average(MIRIonly_g2['mStar']),np.average(MIRIonly_g2['beta']),
      np.average(MIRIonly_g2['SFR_10']),np.average(MIRIonly_g2['SFR_100']),np.average(MIRIonly_g2['tau']))
print('Number of miri sources in group 3: ', len(MIRIonly_g3))
print(np.average(MIRIonly_g3['redshift']),np.average(MIRIonly_g3['mStar']),np.average(MIRIonly_g3['beta']),
      np.average(MIRIonly_g3['SFR_10']),np.average(MIRIonly_g3['SFR_100']),np.average(MIRIonly_g3['tau']))


##################################################################################
### bottom subplot with the average SED
##################################################################################
all_A = []
all_B = []
all_CC = []
## only nircam LW and MIRI
#C_depth = ['darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','crimson','crimson']
for i in np.array([10,11,13,15,17,20,23,24]): # range(len(df_all.iloc[:25])):
    A, = ax_SED.plot(df_all['TC_wl'][i]/10000,df_all['TC'][i]/1000,color=df_all['COLOR'][i],label=df_all['INSTR'][i]) #,label=df['band'][i])
    B = ax_SED.fill_between(df_all['TC_wl'][i]/10000,df_all['TC'][i]/1000, color=df_all['COLOR'][i], alpha=0.5,label=df_all['BAND'][i])
    all_A.append(A)
    all_B.append(B)
    ax_SED.text(df_all['wl'][i]-0.05,0.3e-5,df_all['BAND'][i],color='black',fontsize=6, bbox=dict(facecolor='white',edgecolor='black', boxstyle='square'))
    ##### ADD HERE THE LIMITS I USED TO SELECT THEM
    CC = ax_SED.scatter(df_all['wl'][i], df_all['2.5depth nJy'][i], s=200, marker='_' ,color=df_all['C_DEPTH'][i], label='%s depth'%(df_all['INSTR'][i]) )
    all_CC.append(CC)

second_legend = ax_SED.legend( handles = [all_CC[0],all_CC[-1]],prop={'size': 8},frameon= True,  framealpha=0.7, bbox_to_anchor=(1, 1)) #loc=2,
ax_SED.add_artist(second_legend)

ax_SED.set_ylim(1e-6,1e4)
ax_SED.set_xlim(1e0,1.2e1)
ax_SED.set_xscale('log')
ax_SED.set_yscale('log')
ax_SED.set_xlabel(r'$\lambda_{obs}$ [$\mu$m]',size=20)
ax_SED.set_ylabel(r'f$_\nu^{obs}$ [nJy]',size=20)

############### now add the science i.e. the SED - the median the ~20 individual ones - of the MIRI ONLY 

all_wl_obs_MIRIonly = []
all_s_obs_MIRIonly = []
for i in range( len(fits_r)): # looping on each galaxy in the table 

    ## since we want in OBSERVED frame -> divide the flux by (1+z) and multiply wavelength by (1+z)
    z = fits_r['redshift'][i]

    ### the x axis: wavelength array
    wl_obs = fits_r['WL'][i]*(1+z) ## since i don't give a unit in here i have to give it in u.spectral_density

    ### the y axis: flux array -> convert from u.erg/u.s/u.cm**2/u.AA to nJy and then from RF to OBS frame
    s_cgs_obs = (fits_r['SED'][i]/(1+z)) * u.erg/u.s/u.cm**2/u.AA
    ## convert into nanoJy
    s_nJy_obs = s_cgs_obs.to(u.nJy, u.spectral_density(u.AA, wl_obs) ).value

    ### the x axis: wavelength array
    wl_obs = (wl_obs * u.AA ).to('micron').value # convert it into microns
    # to take then the median 
    all_wl_obs_MIRIonly.append(wl_obs)
    all_s_obs_MIRIonly.append(s_nJy_obs)



#plt.rcParams['image.cmap'] = 'Grays'
#plt.rcParams['figure.figsize'] = (14,10)
#fig.savefig('/Users/irisj/Downloads/SED_goodone.png', dpi=380,format='png')
#fig.savefig('/Users/irisj/Downloads/SED_justSFR.png', dpi=380,format='png')

if show_plots == True:
        plt.show()
else:
    None

fig_SED.subplots_adjust(wspace=0.2, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.15)
fig_SED.savefig('LETTER_MIRIonlyFigure4_SED.png', dpi=380,format='png')


'''
fig_test = plt.figure(figsize=(9,4))
ax_test = fig_test.add_subplot(111)


for i in np.array([10,11,13,15,17,20,23,24]): # range(len(df_all.iloc[:25])):
    A, = ax_test.plot(df_all['TC_wl'][i]/10000,df_all['TC'][i]/1000,color=df_all['COLOR'][i],label=df_all['INSTR'][i]) #,label=df['band'][i])
    B = ax_test.fill_between(df_all['TC_wl'][i]/10000,df_all['TC'][i]/1000, color=df_all['COLOR'][i], alpha=0.5,label=df_all['BAND'][i])
    all_A.append(A)
    all_B.append(B)
    ax_test.text(df_all['wl'][i]-0.05,0.1e-5,df_all['BAND'][i],color='black',fontsize=6, bbox=dict(facecolor='white',edgecolor='black', boxstyle='square'))
    ##### ADD HERE THE LIMITS I USED TO SELECT THEM
    #ax.errorbar(df_all['wl'][i], df_all['2.5depth nJy'][i],fmt='o', capsize=5, color='gray' ) #, color=df_all['COLOR'][i])
    ax_test.scatter(df_all['wl'][i], df_all['2.5depth nJy'][i], s=15, marker='_' ,color=df_all['COLOR'][i] )



ax_test.set_ylim(1e-6,1e4)
ax_test.set_xlim(1e0,1.2e1)
ax_test.set_xscale('log')
ax_test.set_yscale('log')
ax_test.set_xlabel(r'$\lambda_{obs}$ [$\mu$m]',size=20)
ax_test.set_ylabel(r'f$_\nu^{obs}$ [nJy]',size=20)

plt.show()
''' 