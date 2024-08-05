#Generate template spectra spectra from SN_Cosmo, then store in a folder
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
import extinction
import pyarrow.parquet as pq
import pyarrow as pa
import duckdb
import shutil
from astropy.io import fits


def order_of_magnitude(value):
    """
    Function that returns the order of magnitude of a value
    Parameters:
        value - value to find the order of magnitude of
    Returns:
        order of magnitude of the value
    """
    return math.floor(math.log(value, 10))

def generate_model(model_type, wl, redshift = 0.0):
    """
    Function that generates a SN Cosmo model

    Parameters: 
        model_type - SN Cosmo model type
        wl - wavelength range for the model
    Returns: 
        model - SN Cosmo model
        redshift - redshift range for the model
    """
    #Generate the model according to the sn cosmo model type
    model = sncosmo.Model(source = model_type)

    #Set the maximumum allowed redshift
    min_wl = model.minwave()
    max_wl = model.maxwave()
    min_z = (wl[-1]/max_wl) -1
    min_z = 0 if min_z < 0 else min_z
    #max_z = (wl[0]/min_wl) -1 for the maximum redshift allowed by sncosmo
    max_z = 0.6 #for the maximum redshift that will produce usable spectra from 4MOST

    #print(model.maxtime())
    #print(model.mintime())

    if redshift > max_z:
        print("Redshift is greater than the maximum redshift allowed by the model. Redshift has been set to the maximum redshift allowed by the model.")
        redshift = max_z
    elif redshift < min_z:
        print("Redshift is less than the minimum redshift allowed by the model. Redshift has been set to the minimum redshift allowed by the model.")
        redshift = min_z

    #Set the parameters for the model
    model.set(z = redshift)
    return model, redshift, min_wl, max_wl

def sn_spectra_generator(resolution, time_scale, model, time_period, wavelength):
    """
    Function that generates a SN Cosmo model spectrum
    Parameters: 
        resolution - resolution of the spectrum
        time_scale - number of spectra generated/evolution of the spectra
        model - SN Cosmo model
        time_period - time period of the spectra
        wavelength - wavelength range of the spectra
    Returns:
        spectra_evolution - the evolution of the spectra
    """
    spectra_evolution = [[i * resolution + j for j in range(resolution)] for i in range(time_scale)]
    iterable = 0
    for mjd in time_period:
        #Generate the spectrum at the given wavelength and date
        sn_spectrum = model.flux(mjd, wavelength)
        for i in range(len(sn_spectrum)):
            spectra_evolution[iterable][i] = sn_spectrum[i]
        iterable = iterable + 1
    return spectra_evolution

def host_galaxy_spectra_generator(hg_spectra, hg_wavelength, redshift, wl_range, hg_ID = 0):
    """
    Function that takes a host galaxy spectra and rebins and redshifts to that of the supernova
    Parameters:
        hg_spectra - host galaxy spectra
        hg_wavelength - host galaxy wavelength
        redshift - redshift of the supernova
        wl_range - wavelength range of the supernova spectra
    Returns:    
        output_spectra - rebinned and redshifted host galaxy spectra
        output_wl - wavelength of the rebinned and redshifted host galaxy spectra
    """
    #rebin the spectra into 15 Angstrom bins
    full_wl_range = np.arange(wl_range[0], wl_range[-1]+15, 15)
    output_spectra = spectres.spectres(full_wl_range, hg_wavelength, hg_spectra)

    return output_spectra

def redden_spectra(spectra, wavelength):
    """
    Function that reddens a spectra
    Parameters:
        spectra - spectra to be reddened
        wavelength - wavelength range of the spectra
    Returns:
        spectrum_ext - reddened spectra
    """
    ext = G23(Rv = 3.1)
    spectrum_ext = spectra * ext.extinguish(wavelength)
    return spectrum_ext
    
def noisey_sn_spectra_generator(spectra, time):
    """
    Function that adds poisson noise to a 1D spectra
    Parameters:
        spectra - spectra to add noise to
    Returns:
        spectra - spectra with noise added
        time - no of days since explosion
    """
    while True:
        try:
            if not isinstance(spectra, np.ndarray) or spectra.ndim != 1:
                raise Exception("Not a 1D spectra")
            spectra = spectra / np.max(spectra) * 100
            time = 'Unknown'
            break
        except:

            spectra = spectra[time] / np.max(spectra[time]) * 100
            break

    return np.random.poisson(np.clip(spectra, 0, None)), time

def noisey_hg_spectra_generator(hg_spectra):
    final_spectra = hg_spectra / np.max(hg_spectra) * 100
    final_spectra = np.nan_to_num(final_spectra, nan=0)
    return np.random.poisson(np.clip(final_spectra, 0, None))

def create_plot(x_data, y_data, labels, title, x_label, y_label, save_path, save_name):
    """
    Function that creates a single plot given some x_data and y_data
    Parameters:
        x_data - List of x axis data
        y_data - List of y axis data
        labels - List of labels for each data series
        title - Title of the plot
        x_label - x axis label
        y_label - y axis label
        save_path - Path to save the plot
        save_name - Name to save the plot as
    Returns:
        None
    Example Usage:
        x1 = [1, 2, 3, 4, 5]
        y1 = [2, 4, 6, 8, 10]
        label1 = 'Series 1'
        x2 = [1, 2, 3, 4, 5]
        y2 = [1, 2, 1, 2, 1]
        label2 = 'Series 2'
        create_plot([x1, x2], [y1, y2], [label1, label2], 'My Plot', 'X-axis', 'Y-axis', '', 'plot.png')
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    rcParams["axes.linewidth"] = 3
    plt.rcParams.update({"font.size": 13})
    plt.tick_params(axis="both", which="major", width=1.5, length=6, direction="in")
    plt.tick_params(axis="both", which="minor", width=1.5, length=4, direction="in")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.xaxis.set_minor_locator(MultipleLocator((x_data[0][-1] - x_data[0][0]) / 10))
    #ax.yaxis.set_minor_locator(MultipleLocator((y_data[0][-1] - y_data[0][0]) / 10))

    for i, (x, y, label) in enumerate(zip(x_data, y_data, labels)):
        linestyle = '-' if i == 1 else '-' if i == 2 else '-.' if i == 3 else '-'
        color = 'red' if i == 1 else 'blue' if i == 2 else 'green' if i == 3 else 'black'
        plt.plot(x, y, color=color, linestyle=linestyle, label=label)

    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()

def generate_phases(start_time, end_time, time_step):
    """
    Function that generates the phases of a model
    Parameters:
        model - SN Cosmo model
        tau - the array of valid phases
    Returns:
        phases - phases of the model
    """
    tau = np.arange(start_time, end_time, time_step)
    phases = []

    if start_time == 1:
        for i in tau:
            if i > 40:
                if i % 3 == 0:
                    phases.append(i)
                else:
                    continue
            if i < 10:
                if i % 2 == 0:
                    phases.append(i)
                else:
                    continue
            else:
                phases.append(i)


    elif start_time < 0:
        for i in tau:
            if i < -10:
                if i % 2 == 0:
                    phases.append(i)
                else:
                    continue
            elif i > 20:
                if i % 3 == 0:
                    phases.append(i)
                else:
                    continue
            else:
                phases.append(i)
    return phases

 
#------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------Main Code---------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------#

#Generate parameters for the sn cosmo values for 4MOST

template_wl = np.arange(2315, 9015, 15)
spectra_resolution = len(template_wl)

generate_supernova = True
generate_hosts = True
generate_transients = True

pure_sn_dir = 'transient_template_dir'
pure_hg_dir = 'host_template_dir'


os.makedirs(os.path.join(pure_hg_dir), exist_ok=True)
os.makedirs(os.path.join(pure_hg_dir, 'metadata'), exist_ok=True)

#--------------------------------------------------------Generate the host galaxy spectra------------------------------------------------------------

print('---------------------------')
print('Generating Host Galaxies...')
print('---------------------------')

pure_hg_spectra_dict = {}

#Import the host galaxy flux and wavelength and shift to match the redshift to that of the supernova
host_galaxy_directory = 'host_galaxy_gen_dir/hg_spectra_from_NGSF_Dir'
host_galaxy_names = os.listdir(host_galaxy_directory)
number_of_host_galaxies = len(host_galaxy_names)

#Define the dust parameters
extinction_range = np.arange(-.6, .6, 0.2) # Redenning equivalent to ~ -0.2 E(B-V) to 0.2 E(B-V)
r_v = 3.1
dust_law = "ccm89" #Use the CCM89 dust law (Cardelli, Clayton, Mathis 1989)

hg_ID = 1

if generate_hosts == True:
    for hg_i in tqdm(range(number_of_host_galaxies)):
        hg_redshift = 0.0
        for hg_a_v in extinction_range:
            host_galaxy_info = np.loadtxt(os.path.join(host_galaxy_directory, host_galaxy_names[hg_i]))
            host_galaxy_wl = host_galaxy_info[:, 0]
            host_galaxy_flux = host_galaxy_info[:, 1]

            #Trim the spectra to the relevant wavelength range
            pure_hg_spectra = host_galaxy_spectra_generator(host_galaxy_flux, host_galaxy_wl, hg_redshift, template_wl, hg_ID)
            
            # Convert the variables to 'double' data type
            template_wl = template_wl.astype('double')
            pure_hg_spectra = pure_hg_spectra.astype('double')

            #Redden the host galaxy spectra
            dusty_hg_spectra = extinction.apply(extinction.ccm89(template_wl, a_v=hg_a_v, r_v=r_v), pure_hg_spectra)

            hg_mean = np.mean(dusty_hg_spectra)
            norm_hg_spectra = (dusty_hg_spectra / hg_mean * 100)

            #Convert dusty_sn_spectra to 1D array of float 16
            output_hg_spectra = norm_hg_spectra.astype('float16')

            hg_template_file_name = ('phgs_{0}_rs_{1}_{2}.txt'.format(hg_i+1, np.round(hg_redshift, 3), np.round(hg_a_v, 3)))

            hg_parquet_file_path = os.path.join(pure_hg_dir, hg_template_file_name)

            output = np.column_stack((template_wl, output_hg_spectra))

            np.savetxt(os.path.join(pure_hg_dir, hg_template_file_name), output, delimiter=',', fmt='%s')

            pure_hg_spectrum_info = {
                'redshift': hg_redshift,
                'HG Type': host_galaxy_names[hg_i],
                'HG A_v': hg_a_v,
                'path': str(os.path.join(pure_hg_dir))}
            pure_hg_spectra_dict[hg_template_file_name] = pure_hg_spectrum_info
            #np.savetxt(os.path.join(pure_hg_dir, hg_template_file_name), dusty_hg_spectra, delimiter=',', fmt='%s')

pure_hg_spectra_df = pd.DataFrame.from_dict(pure_hg_spectra_dict, orient = 'index')

#--------------------------------------------------------Generate the supernova spectra------------------------------------------------------------

print('---------------------------')
print('Generating Supernova...    ')
print('---------------------------')

pure_sn_spectra_dict = {}

#Remove all the plots in the plots directory to avoid confusion
try:
    shutil.rmtree(os.path.join("python_dir", 'plots'))
except:
    pass

if generate_supernova == True:
    #list of SNCosmo models
    model_names = np.array(['salt3', 'v19-2009kr-corr','v19-2004aw-corr', "v19-2004gv-corr", "nugent-sn91t", "nugent-sn91bg", "nugent-sn1a", "hsiao"])
    sn_types = np.array(['Ia', 'II', 'Ic', 'Ib', "91t-like", "91bg-like", "Ia", "Ia"])
    galactic_source = np.array(['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'])

    salt3_tau = generate_phases(-18, 50, 1) #Ia
    v19_2009kr_corr_tau = generate_phases(-10, 50, 1) #II
    v19_2004aw_corr_tau = generate_phases(-14, 50, 1) #Ic
    v19_2004gv_corr_tau = generate_phases(-20, 50, 1) #Ib
    nugent_sn91t_tau = generate_phases(1, 70, 1) #91t-like
    nugent_sn91bg_tau = generate_phases(1, 70, 1) #91bg-like
    nugent_sn1a_tau = generate_phases(1, 70, 1) #Ia
    hsiao_tau = generate_phases(-18, 50, 1) #Ia

    phases_dict = {
        'salt3': salt3_tau,
        'v19-2009kr-corr': v19_2009kr_corr_tau,
        'v19-2004aw-corr': v19_2004aw_corr_tau,
        'v19-2004gv-corr': v19_2004gv_corr_tau,
        'nugent-sn91t': nugent_sn91t_tau,
        'nugent-sn91bg': nugent_sn91bg_tau,
        'nugent-sn1a': nugent_sn1a_tau,
        'hsiao': hsiao_tau     
}
    
    # Generate directories for the spectra
    for i in sn_types:
        os.makedirs(os.path.join(pure_sn_dir, "type_{0}".format(i)), exist_ok=True)
    os.makedirs(os.path.join(pure_sn_dir, 'metadata'), exist_ok=True)

    number_of_templates = 0

    for i in tqdm(model_names):
        for sn_a_v in extinction_range:
            z_igx = 0.0
            #Generate the model
            salt3_model, redshift, minimum_wl, maximum_wl = generate_model(i, template_wl, z_igx)
            if minimum_wl < 2315:
                minimum_wl = 2315
            else:
                minimum_wl = minimum_wl

            if maximum_wl > 9015:
                maximum_wl = 9015
            else:
                maximum_wl = maximum_wl

            wavelength = np.arange(2315, 9015, 15)
            spectra_resolution = len(wavelength)

            #Set up the phases
            tau = phases_dict[i]  
            
            full_phases = np.arange(-20, 70, 1)
            model_full_range = len(full_phases)
            pure_sn_spectra = np.array((sn_spectra_generator(spectra_resolution, model_full_range, salt3_model, full_phases, wavelength)))
            for tau_idx in tau:
                
                vals_array = np.zeros(len(host_galaxy_names)*len(extinction_range))
                hg_type_igx = 0

                single_sn_spectrum = pure_sn_spectra[int(tau_idx - full_phases[0])]

                # Convert the variables to 'double' data type
                template_wl = template_wl.astype('double')
                wavelength = wavelength.astype('double')
                pure_sn_spectrum = single_sn_spectrum.astype('double')

                #Redden the supernova spectra
                dusty_sn_spectra = extinction.apply(extinction.ccm89(wavelength, a_v=sn_a_v, r_v=r_v), pure_sn_spectrum)

                #Scale up 
                sn_mean = np.mean(dusty_sn_spectra)
                norm_sn_spectra = (dusty_sn_spectra / sn_mean * 100)

                #Convert dusty_sn_spectra to 1D array of float 16
                output_sn_spectra = norm_sn_spectra.astype('float16')

                sn_type_path = ("type_{0}".format(sn_types[np.where(model_names == i)[0][0]]))
                sn_template_file_name = ('{0}_{1}_{2}_{3}.txt'.format(i, tau_idx, np.round(sn_a_v, 2), np.round(z_igx, 2)))


                #Save the wavelength and spectrum to a file
                output = np.column_stack((wavelength, output_sn_spectra))

                np.savetxt(os.path.join(pure_sn_dir, sn_type_path, sn_template_file_name), output, delimiter=',', fmt='%s')

                #Create the plots of the individual spectra at redshift 0 and E(b-v) 0
                if np.round(redshift, 2) == 0.0 and np.round(sn_a_v,2) == 0.0:
                    os.makedirs(os.path.join("python_dir", 'plots', i), exist_ok=True)

                    create_plot([wavelength], [output_sn_spectra], ['Spectra'], '{0} at time = {1}'.format(i, tau_idx), 'Wavelength (Angstroms)', 'Flux', os.path.join("python_dir", 'plots', i), '{0}_{1}'.format(i, tau_idx-tau[0]))

                if tau[0] == 1:
                    t0 = tau_idx
                    tau_idx = "N/A"
                else:
                    t0 = "N/A"
                        
                #Save the pure sn spectra metadata to a dictionary
                pure_sn_spectrum_info = {
                'Model Type: ': i,
                'Type': sn_types[np.where(model_names == i)][0],
                'redshift': redshift,
                'tau': tau_idx,
                't0': t0,
                'SN A_v': sn_a_v,
                'Standard WL': "Y",
                "Galactic Source": galactic_source[np.where(model_names == i)[0][0]],
                'path': str(os.path.join(pure_sn_dir, sn_type_path))}
                pure_sn_spectra_dict[sn_template_file_name] = pure_sn_spectrum_info
                
                number_of_templates = number_of_templates + 1



#--------------------------------------------------------Generate the transient spectra------------------------------------------------------------

print('-----------------------------')
print('Generating Misc Transients...')
print('-----------------------------')
template_wl = np.arange(3800, 8015, 15)

#Generate other transient templates
transient_types = np.array(['CV'])
galactic_source = np.array(['Y'])

for i in transient_types:
    os.makedirs(os.path.join(pure_sn_dir, i), exist_ok=True)

for type in tqdm(transient_types):
    for transient_av in extinction_range:
        transient_names = os.listdir(os.path.join('custom_transients_dir', type))

        for transient_name in transient_names:
            if transient_name == '.DS_Store':
                continue
            transient_path = os.path.join('custom_transients_dir', type, transient_name)

            data= np.loadtxt(str(transient_path), skiprows=1, delimiter='\t')
  
            

            transient_wl = data[:, 0]
            transient_flux = data[:, 1]

            #Trim the spectra to the relevant wavelength range
            trimmed_transient_spectra = spectres.spectres(template_wl, transient_wl, transient_flux, verbose = False)

            nans = np.isnan(trimmed_transient_spectra)
            #get the indices where nans are false
            not_nans = np.where(~nans)[0]
            trimmed_transient_spectra = trimmed_transient_spectra[not_nans]
            template_wl = template_wl[not_nans]

            template_wl = template_wl.astype('double')
            trimmed_transient_spectra = trimmed_transient_spectra.astype('double')

            #Redden the transient spectra
            dusty_transient_spectra = extinction.apply(extinction.ccm89(template_wl, a_v=transient_av, r_v=r_v), trimmed_transient_spectra)

            transient_mean = np.mean(dusty_transient_spectra)
            norm_transient_spectra = (dusty_transient_spectra / transient_mean * 100)

            #Convert dusty_transient_spectra to 1D array of float 16
            output_transient_spectra = norm_transient_spectra.astype('float16')

            transient_template_file_name = ('{0}_{1}_{2}.txt'.format(type, transient_name, np.round(transient_av, 2)))

            output = np.column_stack((template_wl, output_transient_spectra))

            
            np.savetxt(os.path.join(pure_sn_dir, type, transient_template_file_name), output, delimiter=',', fmt='%s')

            tau_idx = "N/A" 
            t0 = "N/A"
            redshift = 0.0

            #Save the transient spectra metadata to a dictionary
            pure_sn_spectrum_info = {
            'Model Type: ': transient_name,
            'Type': type,
            'redshift': redshift,
            'tau': tau_idx,
            't0': t0,
            'SN A_v': transient_av,
            'Standard WL': "N",
            'Galactic Source': galactic_source[np.where(transient_types == type)[0][0]],
            'path': str(os.path.join(pure_sn_dir, type))}
            pure_sn_spectra_dict[transient_template_file_name] = pure_sn_spectrum_info
            
            number_of_templates = number_of_templates + 1



#Turn into a dataframe for exporting
pure_sn_spectra_df = pd.DataFrame.from_dict(pure_sn_spectra_dict, orient = 'index')

#export the names of the spectra 
pure_sn_names = np.array(pure_sn_spectra_df.index.values)
pure_hg_names = np.array(pure_hg_spectra_df.index.values)

np.savetxt(os.path.join(pure_sn_dir, 'metadata', 'pure_sn_spectra_names.txt'), pure_sn_names, delimiter=',', fmt='%s')
np.savetxt(os.path.join(pure_hg_dir, 'metadata', 'pure_hg_spectra_names.txt'), pure_hg_names, delimiter=',', fmt='%s')

#Export the dataframes to a json file
pure_sn_spectra_df.to_json(os.path.join(pure_sn_dir, 'metadata','pure_sn_spectra_data.json'))
pure_hg_spectra_df.to_json(os.path.join(pure_hg_dir, 'metadata', 'pure_hg_spectra_data.json'))


print('---------------------------')
print('    Templates Generated    ')
print('---------------------------')