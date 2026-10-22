Iris Jermann - PhD student DAWN/DTU

Add fluxes in MIRI bands to the JAGUAR simulations.

There are 10 realisations of the catalog so I calculated it for the 10, named JAGUAR_r1_MIRI.fits etc

The fluxes are in nJy and the tables contain SF and Q galaxies. There is a column 'TYPE'

Download the DATA: https://neogal.iap.fr/JAGUAR_mock_catalogue/

The code is: measures_miri_fluxes.py

All the data is (soon) on my computerome under DATA/JAGUAR/





############## notes on my codes

measures_miri_fluxes.py : measures the fluxes in F444W and all the MIRI bands (using python package pyphot)

JAGUAR_SNR2p5.py : find from the catalogue the sources with SNR=2 (NB: for now without correcting anything for the errors), find those average fluxes, compare them to the depth

JAGUAR_MIRIonly_properties.py : extracts catalogs with ID,z,spectra of the MIRI potential in each realisation

JAGUAR_MIRIonly_statistics.py : does science on them like average values etc 


############## notes on my codes : update from july 2024

measures_miri_fluxes.py : measures the fluxes in F444W and all the MIRI bands (using python package pyphot) - i didn't touch that 

JAGUAR_SNR2p5.py : I skip that one since i am not running this and use the i measured in /Users/irisj/Documents/MIRIonly/everything_on_v7_pablo/step1_sextractor/D2_my_proper_depth/depth_sampled_iris_paper.csv

JAGUAR_MIRIonly_properties.py : extracts catalogs with ID,z,spectra of the MIRI potential in each realisation

JAGUAR_MIRIonly_statistics.py : does science on them like average values etc 


############## notes on my codes : update from october 2024

The depth has been updated and finalised with the code /Users/irisj/Documents/GitProjects/depth_code/depth_grid.py 
To be found in tables /Users/irisj/Documents/GitProjects/depth_code/depth_5mag_in_midis.pkl and /Users/irisj/Documents/GitProjects/depth_code/depth_10nJy_in_midis.pkl

I decide to take 2.5 time this value in 10nJy that I measure There

Let's then run these 2 codes:

1. JAGUAR_MIRIonly_properties.py : extracts catalogs with ID,z,spectra of the MIRI potential in each realisation. Spectra only extracted for R1.

2. JAGUAR_MIRIonly_statistics.py : does science on them like average values etc 