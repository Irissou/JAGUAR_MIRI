###### IRIS JERMANN PhD Student DAWN/DTU

'''
This code measures the fluxes in MIRI bands of the JAGUAR simulation.
And returns in one Table everything (except the spectra).
I compute F444W as verification and the MIRI F560W, F770W, F1000W, and all the other ones as well!!

NB: the JAGUAR spectra are restframe and in erg/s/cm^2/A and the pyphot thing returns it in the same unit.
The unit of the passband doens't matter unless we want any electron related unit.
The fluxes in bands from the JAGUAR tables are in nanoJy.

-> the table I save has the fluxes in nJy
'''


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from collections import Counter

import astropy.cosmology as cosmo
from astropy.table import Table, vstack, hstack


from astropy.io import fits
import astropy.units as u

import pandas as pd
import pyphot

from pyphot import unit, Vega, Filter


##################################################################################
#### DEFINE HERE WHICH REALISATIONs
##################################################################################

#R = 'r3'

#for R in ['r4','r5','r6','r7','r8','r9','r10']:

##################################################################################
#### IMPORT THE TRANSMISSION CURVES AND THE REF WAVELENGTH FOR EACH BAND
##################################################################################

lamb_eff_F444W = 44043.15 * u.AA
lamb_eff_F560W = 56352.56 * u.AA 
lamb_eff_F770W = 76393.34 * u.AA 
lamb_eff_F1000W = 99531.16 * u.AA 

lamb_eff_F1130W = 113085.01 * u.AA 
lamb_eff_F1280W = 128101.38 * u.AA 
lamb_eff_F1500W = 150635.06 * u.AA 
lamb_eff_F1800W = 179837.22 * u.AA 
lamb_eff_F2100W = 207950.05 * u.AA 
lamb_eff_F2550W = 253640.02 * u.AA 

wl_F444W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/NIRCam/JWST_NIRCam.F444W.dat')[:,0] #* u.AA
tc_F444W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/NIRCam/JWST_NIRCam.F444W.dat')[:,1] 

wl_F560W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F560W.dat')[:,0] #* u.AA
tc_F560W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F560W.dat')[:,1] 

wl_F770W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F770W.dat')[:,0] #* u.AA
tc_F770W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F770W.dat')[:,1] 

wl_F1000W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1000W.dat')[:,0] #* u.AA
tc_F1000W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1000W.dat')[:,1] 

wl_F1130W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1130W.dat')[:,0] #* u.AA
tc_F1130W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1130W.dat')[:,1] 

wl_F1280W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1280W.dat')[:,0] #* u.AA
tc_F1280W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1280W.dat')[:,1] 

wl_F1500W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1500W.dat')[:,0] #* u.AA
tc_F1500W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1500W.dat')[:,1] 

wl_F1800W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1800W.dat')[:,0] #* u.AA
tc_F1800W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F1800W.dat')[:,1] 

wl_F2100W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F2100W.dat')[:,0] #* u.AA
tc_F2100W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F2100W.dat')[:,1] 

wl_F2550W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F2550W.dat')[:,0] #* u.AA
tc_F2550W = np.loadtxt('/Users/irisj/software/bandpass_plots/JWST/MIRI/JWST_MIRI.F2550W.dat')[:,1] 



f_F444W  = Filter(wl_F444W, tc_F444W, name='JWST_MIRI_F444W', dtype='photon', unit='Angstrom')
f_F560W  = Filter(wl_F560W, tc_F560W, name='JWST_MIRI_F560W', dtype='photon', unit='Angstrom')
f_F770W  = Filter(wl_F770W, tc_F770W, name='JWST_MIRI_F770W', dtype='photon', unit='Angstrom')
f_F1000W  = Filter(wl_F1000W, tc_F1000W, name='JWST_MIRI_F1000W', dtype='photon', unit='Angstrom')
f_F1130W  = Filter(wl_F1130W, tc_F1130W, name='JWST_MIRI_F1130W', dtype='photon', unit='Angstrom')
f_F1280W  = Filter(wl_F1280W, tc_F1280W, name='JWST_MIRI_F1280W', dtype='photon', unit='Angstrom')
f_F1500W  = Filter(wl_F1500W, tc_F1500W, name='JWST_MIRI_F1500W', dtype='photon', unit='Angstrom')
f_F1800W  = Filter(wl_F1800W, tc_F1800W, name='JWST_MIRI_F1800W', dtype='photon', unit='Angstrom')
f_F2100W  = Filter(wl_F2100W, tc_F2100W, name='JWST_MIRI_F2100W', dtype='photon', unit='Angstrom')
f_F2550W  = Filter(wl_F2550W, tc_F2550W, name='JWST_MIRI_F2550W', dtype='photon', unit='Angstrom')

##################################################################################
#### EVERYTHING FOR THE QUIESCENT GALAXIES
##################################################################################


for R in ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10']:

    ## local path 
    # Q_cat_all = Table.read('/Users/irisj/software/JAGUAR/%s/JADES_Q_mock_%s_v1.2.fits'%(R,R))
    # Q_cat_all['SPECTRA'] = np.array( fits.open('/Users/irisj/software/JAGUAR/%s/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits'%(R,R))[1].data)
    # Q_cat_all['WL'] = np.array([fits.open('/Users/irisj/software/JAGUAR/%s/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits'%(R,R))[2].data] * len(Q_cat_all['SPECTRA']))

    ## harddrive path /Volumes/champignon/JAGUAR 
    Q_cat_all = Table.read('/Volumes/champignon/JAGUAR/%s/JADES_Q_mock_%s_v1.2.fits'%(R,R))
    Q_cat_all['SPECTRA'] = np.array( fits.open('/Volumes/champignon/JAGUAR/%s/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits'%(R,R))[1].data)
    Q_cat_all['WL'] = np.array([fits.open('/Volumes/champignon/JAGUAR/%s/JADES_Q_mock_%s_v1.2_spec_5A_30um.fits'%(R,R))[2].data] * len(Q_cat_all['SPECTRA']))


    ## for K correction and redshift
    Q_one_plus_z = np.tile(np.array((1+Q_cat_all['redshift'])),(len(Q_cat_all['WL'][0]),1)).T


    fluxes_F444W = f_F444W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F560W = f_F560W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F770W = f_F770W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F1000W = f_F1000W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F1130W = f_F1130W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F1280W = f_F1280W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F1500W = f_F1500W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F1800W = f_F1800W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F2100W = f_F2100W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    fluxes_F2550W = f_F2550W.get_flux( Q_cat_all['WL']*Q_one_plus_z,
                            Q_cat_all['SPECTRA']/Q_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

    Q_fluxes_F444W_nJy = fluxes_F444W.to(u.nJy, u.spectral_density( lamb_eff_F444W) ).value
    Q_fluxes_F560W_nJy = fluxes_F560W.to(u.nJy, u.spectral_density( lamb_eff_F560W) ).value
    Q_fluxes_F770W_nJy = fluxes_F770W.to(u.nJy, u.spectral_density( lamb_eff_F770W) ).value
    Q_fluxes_F1000W_nJy = fluxes_F1000W.to(u.nJy, u.spectral_density( lamb_eff_F1000W) ).value

    Q_fluxes_F1130W_nJy = fluxes_F1130W.to(u.nJy, u.spectral_density( lamb_eff_F1130W) ).value
    Q_fluxes_F1280W_nJy = fluxes_F1280W.to(u.nJy, u.spectral_density( lamb_eff_F1280W) ).value
    Q_fluxes_F1500W_nJy = fluxes_F1500W.to(u.nJy, u.spectral_density( lamb_eff_F1500W) ).value
    Q_fluxes_F1800W_nJy = fluxes_F1800W.to(u.nJy, u.spectral_density( lamb_eff_F1800W) ).value
    Q_fluxes_F2100W_nJy = fluxes_F2100W.to(u.nJy, u.spectral_density( lamb_eff_F2100W) ).value
    Q_fluxes_F2550W_nJy = fluxes_F2550W.to(u.nJy, u.spectral_density( lamb_eff_F2550W) ).value




    ##################################################################################
    #### EVERYTHING FOR THE STAR FORMING GALAXIES
    # here since the spectra are heavy and take lots of space, will have to
    # load or split the stuff
    ##################################################################################



    SF_cat_all = Table.read('/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2.fits'%(R,R))

    SF_spectra_part1 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_0p2_1.fits'%(R,R)
    SF_spectra_part2 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_1_1p5.fits'%(R,R)
    SF_spectra_part3 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_1p5_2.fits'%(R,R)
    SF_spectra_part4 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_2_3.fits'%(R,R)
    SF_spectra_part5 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_3_4.fits'%(R,R)
    SF_spectra_part6 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_4_5.fits'%(R,R)
    SF_spectra_part7 = '/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2_spec_5A_30um_z_5_15.fits'%(R,R)

    SF_name_list = [SF_spectra_part1, SF_spectra_part2, SF_spectra_part3, SF_spectra_part4, SF_spectra_part5, SF_spectra_part6, SF_spectra_part7]

    SF_ALL_F444W_nJy = np.array([])
    SF_ALL_F560W_nJy = np.array([])
    SF_ALL_F770W_nJy = np.array([])
    SF_ALL_F1000W_nJy = np.array([])
    SF_ALL_F1130W_nJy = np.array([])
    SF_ALL_F1280W_nJy = np.array([])
    SF_ALL_F1500W_nJy = np.array([])
    SF_ALL_F1800W_nJy = np.array([])
    SF_ALL_F2100W_nJy = np.array([])
    SF_ALL_F2550W_nJy = np.array([])



    ### open them in a loop and make maxi list of wl and spectra
    for i in range(len(SF_name_list)):
        this_name = SF_name_list[i]

        this_SPECTRA = np.array( fits.open(this_name)[1].data)
        this_WL = np.array([fits.open(this_name)[2].data] * len(this_SPECTRA))
        this_z = fits.open(this_name)[3].data['redshift']

        
        SF_one_plus_z = np.tile(np.array((1+this_z)),(len(this_WL[0]),1)).T

        fluxes_F444W = f_F444W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

        fluxes_F560W = f_F560W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

        fluxes_F770W = f_F770W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

        fluxes_F1000W = f_F1000W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA
        
        fluxes_F1130W = f_F1130W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA
        
        fluxes_F1280W = f_F1280W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA
        
        fluxes_F1500W = f_F1500W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA
        
        fluxes_F1800W = f_F1800W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA
        
        fluxes_F2100W = f_F2100W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA
        
        fluxes_F2550W = f_F2550W.get_flux( this_WL*SF_one_plus_z,
                                this_SPECTRA/SF_one_plus_z, axis=1)* u.erg/u.s/u.cm**2/u.AA

        this_SF_fluxes_F444W_nJy = fluxes_F444W.to(u.nJy, u.spectral_density( lamb_eff_F444W) ).value
        this_SF_fluxes_F560W_nJy = fluxes_F560W.to(u.nJy, u.spectral_density( lamb_eff_F560W) ).value
        this_SF_fluxes_F770W_nJy = fluxes_F770W.to(u.nJy, u.spectral_density( lamb_eff_F770W) ).value
        this_SF_fluxes_F1000W_nJy = fluxes_F1000W.to(u.nJy, u.spectral_density( lamb_eff_F1000W) ).value
        this_SF_fluxes_F1130W_nJy = fluxes_F1130W.to(u.nJy, u.spectral_density( lamb_eff_F1130W) ).value
        this_SF_fluxes_F1280W_nJy = fluxes_F1280W.to(u.nJy, u.spectral_density( lamb_eff_F1280W) ).value
        this_SF_fluxes_F1500W_nJy = fluxes_F1500W.to(u.nJy, u.spectral_density( lamb_eff_F1500W) ).value
        this_SF_fluxes_F1800W_nJy = fluxes_F1800W.to(u.nJy, u.spectral_density( lamb_eff_F1800W) ).value
        this_SF_fluxes_F2100W_nJy = fluxes_F2100W.to(u.nJy, u.spectral_density( lamb_eff_F2100W) ).value
        this_SF_fluxes_F2550W_nJy = fluxes_F2550W.to(u.nJy, u.spectral_density( lamb_eff_F2550W) ).value

        SF_ALL_F444W_nJy = np.concatenate((SF_ALL_F444W_nJy, this_SF_fluxes_F444W_nJy))
        SF_ALL_F560W_nJy = np.concatenate((SF_ALL_F560W_nJy, this_SF_fluxes_F560W_nJy))
        SF_ALL_F770W_nJy = np.concatenate((SF_ALL_F770W_nJy, this_SF_fluxes_F770W_nJy))
        SF_ALL_F1000W_nJy = np.concatenate((SF_ALL_F1000W_nJy, this_SF_fluxes_F1000W_nJy))
        SF_ALL_F1130W_nJy = np.concatenate((SF_ALL_F1130W_nJy, this_SF_fluxes_F1130W_nJy))
        SF_ALL_F1280W_nJy = np.concatenate((SF_ALL_F1280W_nJy, this_SF_fluxes_F1280W_nJy))
        SF_ALL_F1500W_nJy = np.concatenate((SF_ALL_F1500W_nJy, this_SF_fluxes_F1500W_nJy))
        SF_ALL_F1800W_nJy = np.concatenate((SF_ALL_F1800W_nJy, this_SF_fluxes_F1800W_nJy))
        SF_ALL_F2100W_nJy = np.concatenate((SF_ALL_F2100W_nJy, this_SF_fluxes_F2100W_nJy))
        SF_ALL_F2550W_nJy = np.concatenate((SF_ALL_F2550W_nJy, this_SF_fluxes_F2550W_nJy))




    print(SF_ALL_F1000W_nJy)
    print(np.shape(SF_ALL_F1000W_nJy))


    ##################################################################################
    #### CONCAT MAXI TABLE WITH ALL OF THEM, ALSO ADD A COLUMN WITH THE TYPE OF IT
    ##################################################################################
    # using vstack or hstack or df something -> actually I can't for the spectra since they don't have the same shapes 
    # but i can do for the general properties

    Q_cat_all_stack = Table.read('/Volumes/champignon/JAGUAR/%s/JADES_Q_mock_%s_v1.2.fits'%(R,R))
    Q_cat_all_stack['TYPE'] = len(Q_cat_all_stack)*['QG']
    Q_cat_all_stack['F444W'] = Q_fluxes_F444W_nJy
    Q_cat_all_stack['F560W'] = Q_fluxes_F560W_nJy
    Q_cat_all_stack['F770W'] = Q_fluxes_F770W_nJy
    Q_cat_all_stack['F1000W'] = Q_fluxes_F1000W_nJy
    Q_cat_all_stack['F1130W'] = Q_fluxes_F1130W_nJy
    Q_cat_all_stack['F1280W'] = Q_fluxes_F1280W_nJy
    Q_cat_all_stack['F1500W'] = Q_fluxes_F1500W_nJy
    Q_cat_all_stack['F1800W'] = Q_fluxes_F1800W_nJy
    Q_cat_all_stack['F2100W'] = Q_fluxes_F2100W_nJy
    Q_cat_all_stack['F2550W'] = Q_fluxes_F2550W_nJy

    SF_cat_all_stack = Table.read('/Volumes/champignon/JAGUAR/%s/JADES_SF_mock_%s_v1.2.fits'%(R,R))
    SF_cat_all_stack['TYPE'] = len(SF_cat_all_stack)*['SFR']
    SF_cat_all_stack['F444W'] = SF_ALL_F444W_nJy
    SF_cat_all_stack['F560W'] = SF_ALL_F560W_nJy
    SF_cat_all_stack['F770W'] = SF_ALL_F770W_nJy
    SF_cat_all_stack['F1000W'] = SF_ALL_F1000W_nJy
    SF_cat_all_stack['F1130W'] = SF_ALL_F1130W_nJy
    SF_cat_all_stack['F1280W'] = SF_ALL_F1280W_nJy
    SF_cat_all_stack['F1500W'] = SF_ALL_F1500W_nJy
    SF_cat_all_stack['F1800W'] = SF_ALL_F1800W_nJy
    SF_cat_all_stack['F2100W'] = SF_ALL_F2100W_nJy
    SF_cat_all_stack['F2550W'] = SF_ALL_F2550W_nJy


    cat_all = vstack([SF_cat_all_stack,Q_cat_all_stack])

    #cat_all.write('Q_SF_combined_MIRI.fits', overwrite = True)

    cat_all.write('/Volumes/champignon/JAGUAR/Outputs/%s/JAGUAR_%s_MIRI.fits'%(R,R), overwrite = True)

    print('REALISATION %s DONE.'%R)










'''
### some old code just to not loose it
f_F444W  = Filter(wl_F444W, tc_F444W, name='JWST_MIRI_F444W', dtype='photon', unit='Angstrom')
## this is the REF wavelength, it is what is the closest to the results from their table
lamb_eff_F444W = 44043.15 * u.AA
lamb_eff_F560W = 56352.56 * u.AA 
lamb_eff_F770W = 76393.34 * u.AA 
lamb_eff_F1000W = 99531.16 * u.AA 
f_F560W  = Filter(wl_F560W, tc_F560W, name='JWST_MIRI_F560W', dtype='photon', unit='Angstrom')
f_F770W  = Filter(wl_F770W, tc_F770W, name='JWST_MIRI_F770W', dtype='photon', unit='Angstrom')
f_F1000W  = Filter(wl_F1000W, tc_F1000W, name='JWST_MIRI_F1000W', dtype='photon', unit='Angstrom')

def MeasureFlux_with_z(filt, wl_sed, sed, z):
    #the spectra from JAGUAR are in erg/s/cm^2/A (RESTFRAME, so if want observed need to divide by (1+z))
    #the fluxes in the band are in nJy (observed flux density(?))
    #so -> wl_sed and sed must be in function the RF
    
    flux = filt.get_flux(wl_sed*(1+z), sed/(1+z))
    return flux * u.erg/u.s/u.cm/u.cm/u.AA # *unit['erg/s/cm**2/AA']

def MeasureFlux(filt, wl_sed, sed):
    
    #the spectra from JAGUAR are in erg/s/cm^2/A (RESTFRAME, so if want observed need to divide by (1+z))
    #the fluxes in the band are in nJy (observed flux density(?))
    #so -> wl_sed and sed must be in OBSERVED FRAME so already the redshift taken out
    flux = filt.get_flux(wl_sed, sed)
    return flux * u.erg/u.s/u.cm**2/u.AA

for i in range(3):
    this_wl = Q_cat_all['WL'][i]
    this_sed = Q_cat_all['SPECTRA'][i]
    this_z =  Q_cat_all['redshift'][i]
    this_F444W = MeasureFlux_with_z(f_F444W, this_wl,this_sed,this_z)
    this_F560W = MeasureFlux_with_z(f_F560W, this_wl,this_sed,this_z)
    this_F770W = MeasureFlux_with_z(f_F770W, this_wl,this_sed,this_z)
    this_F1000W = MeasureFlux_with_z(f_F1000W, this_wl,this_sed,this_z)
    print('Flux in F444W from the table observed and in nanoJy: ', Q_cat_all['NRC_F444W_fnu'][i])
    print('Flux in F444W: ', this_F444W.to(u.nJy, u.spectral_density( lamb_eff_F444W) )  ) 
    ## here either u.spectral_density(lamb_eff_F444W) where lambeff has a unit or u.spectral_density(u.AA, lamb_eff_F444W) )
    print('Flux in F560W: ', this_F560W.to(u.nJy, u.spectral_density( lamb_eff_F560W 	 ) ))  
    print('Flux in F770W: ', this_F770W.to(u.nJy, u.spectral_density( lamb_eff_F770W 	) )  ) 
    print('Flux in F1000W: ', this_F1000W.to(u.nJy, u.spectral_density( lamb_eff_F1000W ) )   )

### this measures the flux
#flux_band = f_F560W.get_flux(f_F560W, flux_spectrum)

#mags = -2.5 * np.log10(flux_band) - f.AB_zero_mag

#flux_band_jy = flux_band.to('Jy') #1.E6*10.**(-0.4*(mags-8.90))   # [uJy]
'''