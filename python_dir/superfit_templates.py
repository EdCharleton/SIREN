import sncosmo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import math
import os
import pandas as pd
from tqdm import tqdm
import astropy
from astropy import units as u
from  dust_extinction.parameter_averages import G23
import sys
import spectres
from os.path import splitext
import extinction

#import all the files from the sne_superfit folder, where each folder is a different supernova type and each file is a different supernova

#path to the folder containing all the supernova folders
path = "sne_superfit"

#list of all the supernova folders 
sn_types = [folder for folder in os.listdir(path) if not folder.startswith('.')]
redshift_range = np.arange(0, 0.6, 0.01)
extinction_range = np.arange(-.1, .5, 0.1)

#create new directories with the templates for each supernova type
sn_template_dir = "sn_template_dir"
if not os.path.exists(sn_template_dir):
    #Delete durrent directory and everything in it if it exists

    os.makedirs(sn_template_dir)
    os.makedirs(sn_template_dir + "/metadata")
    for i in sn_types:
        os.makedirs(sn_template_dir + "/" + i)

superfit_files_dict = {}

for sn_type_idx in tqdm(sn_types):
    #path to the supernova folder
    sn_type_path = path + "/" + sn_type_idx
    #list of all the supernova files
    sns = [sn for sn in os.listdir(sn_type_path) if not sn.startswith('.')]
    
    #loop through each supernova file
    for sn_idx in tqdm(sns):
        #path to the supernova file
        sn_path = sn_type_path + "/" + sn_idx
        #read in the supernova file
        sn = np.loadtxt(sn_path)
        phase_name = splitext(sn_idx)[0]
        phase_code = splitext(phase_name)[1]
        if phase_code[1] == "p":
            phase = int(phase_code[2:])
        elif phase_code[1] == "m" and phase_code[2] == "0":
            phase = -1 * int(phase_code[2:])
        elif phase_code[1] == "u":
            phase = "Unknown"
        elif phase_code[1:] == "max":
            phase = 0    
        else:
            phase = int(phase_code)

        sn_wavelength = sn[:,0]
        sn_spectrum = sn[:,1]
        sn_error = sn[:,2]

        wl_resolution = np.arange(sn_wavelength[0], sn_wavelength[-1], 15)

        try:
            new_res_spectrum = spectres.spectres(wl_resolution, sn_wavelength, sn_spectrum, verbose = False)
        except:
            print("Error in spectres")
            continue
        
        #Redshift the spectrum
        for redshift in redshift_range:
            z_wl = (1 + redshift) * wl_resolution

            #Redden the spectrum
            for eb_v in extinction_range:

                new_res_spectrum.astype("double")
                z_wl.astype("double")

                extincted_spectrum = extinction.apply(extinction.ccm89(z_wl, a_v=eb_v, r_v=3.1), new_res_spectrum)

                #save the spectrum and the wavelength to a file:
                file_name = sn_template_dir + "/" + sn_type_idx + "/" + phase_name + "_z" + str(redshift) + "_ebv" + str(eb_v) + ".txt"
                np.savetxt(file_name, np.column_stack((z_wl, extincted_spectrum)), fmt='%1.3f %1.7e')

                #save the meta data to a dictionary

                sn_spectrum_info = {
                'Type': sn_type_idx,
                'redshift': redshift,
                'tau': phase,
                'SN E(B-V)': eb_v,
                'path': file_name
                }
                superfit_files_dict[file_name] = sn_spectrum_info

superfit_spectra_df = pd.DataFrame.from_dict(superfit_files_dict, orient = 'index')

sn_filenames = np.array(superfit_spectra_df.index.values)
np.savetxt(os.path.join(sn_template_dir, 'metadata', 'superfit_template_filenames.txt'), sn_filenames, delimiter=',', fmt='%s')
superfit_spectra_df.to_json(os.path.join(sn_template_dir, 'metadata','superfit_sn_template_metadata.json'))


            






        
        
       