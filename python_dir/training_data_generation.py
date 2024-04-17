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

print('---------------------------')
print('Generating host galaxies...')
print('---------------------------')

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

    if redshift > max_z:
        print("Redshift is greater than the maximum redshift allowed by the model. Redshift has been set to the maximum redshift allowed by the model.")
        redshift = max_z
    elif redshift < min_z:
        print("Redshift is less than the minimum redshift allowed by the model. Redshift has been set to the minimum redshift allowed by the model.")
        redshift = min_z

    #Set the parameters for the model
    model.set(z = redshift)
    return model, redshift

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
    #rebin the spectra into 20 Angstrom bins
    full_wl_range = np.arange(wl_range[0]-1520, wl_range[-1]+50, 20)
    output_spectra = spectres.spectres(full_wl_range, hg_wavelength, hg_spectra)
    #Redshift the spectra

    rs_hg_wl = full_wl_range * (1 + redshift)
    #Rebin the spectra to the same wavelength range as 4MOST
    rs_output_spectra = spectres.spectres(wl_range, rs_hg_wl, output_spectra)

    
    return rs_output_spectra

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
#------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------Main Code---------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------#

#Generate parameters for the sn cosmo values for 4MOST
tau = np.linspace(-20, 50, 71)
length_of_phase = len(tau)
four_most_wl = np.arange(3790, 9530, 20)
spectra_resolution = len(four_most_wl)

model_names = np.array(['salt3', 'v19-2009kr-corr','v19-2004aw-corr', "v19-2004gv-corr"])
sn_types = np.array(['Ia', 'II', 'Ic', 'Ib'])


#model_names = np.array(['salt3'])
#sn_types = np.array(['Ia'])

pure_sn_dir = 'pure_sn_spectra_dir'
noisey_sn_dir = 'noisey_sn_spectra_dir'
pure_hg_dir = 'pure_hg_spectra_dir'
noisey_hg_dir = 'noisey_hg_spectra_dir'
convolved_spectra_dir = 'convolved_spectra_dir'

# Generate directories for the spectra
for i in sn_types:
    os.makedirs(os.path.join(pure_sn_dir, "type_{0}".format(i)), exist_ok=True)
    os.makedirs(os.path.join(noisey_sn_dir, "type_{0}".format(i)), exist_ok=True)
    os.makedirs(os.path.join(pure_hg_dir), exist_ok=True)
    os.makedirs(os.path.join(noisey_hg_dir), exist_ok=True)
    os.makedirs(os.path.join(convolved_spectra_dir), exist_ok=True)

pure_hg_spectra_dict = {}
noisey_hg_spectra_dict = {}

#Import the host galaxy flux and wavelength and shift to match the redshift to that of the supernova
host_galaxy_directory = 'host_galaxy_gen_dir/hg_spectra_from_NGSF_Dir'
host_galaxy_names = os.listdir(host_galaxy_directory)
number_of_host_galaxies = len(host_galaxy_names)

#Define the dust parameters
extinction_range = np.arange(-0.2, 1, 0.1)
r_v = 3.1
dust_law = "ccm89" #Use the CCM89 dust law (Cardelli, Clayton, Mathis 1989)

hg_ID = 1
for hg_i in tqdm(range(number_of_host_galaxies)):
    for hg_redshift in np.arange(0, 0.6, 0.01):
        for hg_a_v in extinction_range:
            host_galaxy_info = np.loadtxt(os.path.join(host_galaxy_directory, host_galaxy_names[hg_i]))
            host_galaxy_wl = host_galaxy_info[:, 0]
            host_galaxy_flux = host_galaxy_info[:, 1]

            #Trim the spectra to the relevant wavelength range
            pure_hg_spectra = host_galaxy_spectra_generator(host_galaxy_flux, host_galaxy_wl, hg_redshift, four_most_wl, hg_ID)
            
            # Convert the variables to 'double' data type
            four_most_wl = four_most_wl.astype('double')
            noisey_hg_spectra = pure_hg_spectra.astype('double')

            #Redden the host galaxy spectra
            dusty_hg_spectra = extinction.apply(extinction.ccm89(four_most_wl, a_v=hg_a_v, r_v=r_v), noisey_hg_spectra)

            #Add noise to the spectrum
            noisey_hg_spectra = noisey_hg_spectra_generator(dusty_hg_spectra)

            pure_hg_spectrum_info = {
                'redshift': hg_redshift,
                'HG Type': host_galaxy_names[hg_i],
                'HG A_v': hg_a_v,
                'path': str(os.path.join(pure_hg_dir))}
            pure_hg_spectra_dict['phgs_{0}_rs_{1}_{2}.txt'.format(hg_i+1, np.round(hg_redshift, 2), hg_a_v)] = pure_hg_spectrum_info
            np.savetxt(os.path.join(pure_hg_dir, 'phgs_{0}_rs_{1}_{2}.txt'.format(hg_i+1, np.round(hg_redshift, 2), hg_a_v)), dusty_hg_spectra, delimiter=',', fmt='%s')

            noisey_hg_spectrum_info = {
                'redshift': hg_redshift,
                'HG Type': host_galaxy_names[hg_i],
                'HG A_v': hg_a_v,
                'path': str(os.path.join(noisey_hg_dir))}
            noisey_hg_spectra_dict['nhgs_{0}_rs_{1}_{2}.txt'.format(hg_i+1, np.round(hg_redshift, 2), hg_a_v)] = noisey_hg_spectrum_info
            np.savetxt(os.path.join(noisey_hg_dir, 'nhgs_{0}_rs_{1}_{2}.txt'.format(hg_i+1, np.round(hg_redshift, 2), hg_a_v)), noisey_hg_spectra, delimiter=',', fmt='%s')
            hg_ID = hg_ID + 1


number_of_spectra = 1
redshift_range = np.arange(0, 0.6, 0.01)
pure_sn_spectra_dict = {}
noisey_sn_spectra_dict = {}
convolved_spectra_dict = {}

print('---------------------------')
print('Generating training data...')
print('---------------------------')

pure_hg_spectra_df = pd.DataFrame.from_dict(pure_hg_spectra_dict, orient = 'index')
noisey_hg_spectra_df = pd.DataFrame.from_dict(noisey_hg_spectra_dict, orient = 'index')

number_of_templates = 0
for i in tqdm(model_names):
    for tau_idx in tqdm(tau):
        for sn_a_v in extinction_range:
            for z_igx in redshift_range:
                #Generate the model
                salt3_model, redshift = generate_model(i, four_most_wl, z_igx)

                pure_sn_spectra = np.array((sn_spectra_generator(spectra_resolution, length_of_phase, salt3_model, tau, four_most_wl)))

                noisey_sn_spectra, day = noisey_sn_spectra_generator(pure_sn_spectra, int(tau_idx - tau[0]))
                day = int(day + tau[0])
                
                vals_array = np.zeros(len(host_galaxy_names)*len(extinction_range)*len(redshift_range))
                hg_type_igx = 0

                #Search the dataframe of the host galaxy spectra and import the one with the right redshift
                for n in range(len(noisey_hg_spectra_df['redshift'])):
                    if noisey_hg_spectra_df['redshift'].iloc[n] == redshift:
                        vals_array[hg_type_igx] = n
                        hg_type_igx = hg_type_igx + 1

                #Select a random host galaxy spectra to create our fake spectrum
                val = int(np.random.choice(vals_array[vals_array != 0]))

                noisey_hg_file_names = noisey_hg_spectra_df.index.values
                noisey_hg_spectra = np.loadtxt(os.path.join(noisey_hg_spectra_df['path'].iloc[val], noisey_hg_file_names[val]), delimiter=',')
                pure_hg_spectra = np.loadtxt(os.path.join(pure_hg_spectra_df['path'].iloc[val], pure_hg_spectra_df.index.values[val]), delimiter=',')
            
                #Add the host galaxy spectra to the supernova spectra
                contamination_amount = np.random.uniform(0.1, 2)
                normalised_noisey_hg_spectra = noisey_hg_spectra * contamination_amount

                # Convert the variables to 'double' data type
                four_most_wl = four_most_wl.astype('double')
                noisey_sn_spectra = noisey_sn_spectra.astype('double')
                normalised_noisey_hg_spectra = normalised_noisey_hg_spectra.astype('double')

                #Redden the supernova spectra
                dusty_sn_spectra = extinction.apply(extinction.ccm89(four_most_wl, a_v=sn_a_v, r_v=r_v), noisey_sn_spectra)

                #Redden the host galaxy spectra
                dusty_hg_spectra = extinction.apply(extinction.ccm89(four_most_wl, a_v=hg_a_v, r_v=r_v), normalised_noisey_hg_spectra)

                #Add the host galaxy spectra to the supernova spectra
                convolved_spectra = dusty_sn_spectra + dusty_hg_spectra

                #Save the noisey sn spectra metadata to a dictionary
                noisey_sn_spectrum_info = {
                'Model Type: ': i,
                'Type': sn_types[np.where(model_names == i)],
                'redshift': redshift,
                'tau': day,
                'SN A_v': sn_a_v,
                'path': str(os.path.join(noisey_sn_dir, 'type_{0}'.format(sn_types[np.where(model_names == i)[0][0]])))}
                noisey_sn_spectra_dict['{0}_{1}_{2}_{3}.txt'.format(i, tau_idx, np.round(sn_a_v, 2), np.round(z_igx, 2))] = noisey_sn_spectrum_info

                #Save the pure sn spectra metadata to a dictionary
                pure_sn_spectrum_info = {
                'Model Type: ': i,
                'Type': sn_types[np.where(model_names == i)],
                'redshift': redshift,
                'tau': day,
                'SN A_v': sn_a_v,
                'path': str(os.path.join(pure_sn_dir, 'type_{0}'.format(sn_types[np.where(model_names == i)[0][0]])))}
                pure_sn_spectra_dict['{0}_{1}_{2}_{3}.txt'.format(i, tau_idx, np.round(sn_a_v, 2), np.round(z_igx, 2))] = pure_sn_spectrum_info
                
                #Save the convolved spectra metadata to a dictionary
                convolved_spectrum_info = {
                    'Model Type: ': i,
                    'Type': sn_types[np.where(model_names == i)],
                    'redshift': redshift,
                    'tau': day,
                    'host galaxy contamination': contamination_amount,
                    'HG Type': noisey_hg_spectra_df['HG Type'].iloc[val],
                    'SN A_v': sn_a_v,
                    'HG A_v': noisey_hg_spectra_df['HG A_v'].iloc[val],
                    'R_V': r_v,
                    'Dust Law': dust_law,
                    'path': str(os.path.join(convolved_spectra_dir))}
                convolved_spectra_dict['{0}_{1}_{2}_{3}_{4}'.format(i, tau_idx, noisey_hg_file_names[val], np.round(sn_a_v, 2), np.round(z_igx, 2))] = convolved_spectrum_info

                #Export the individual spectra to a txt file
                np.savetxt(os.path.join(pure_sn_dir, "type_{0}".format(sn_types[np.where(model_names == i)[0][0]]), '{0}_{1}_{2}_{3}.txt'.format(i, tau_idx, np.round(sn_a_v, 2), np.round(z_igx, 2))), pure_sn_spectra[day], delimiter=',', fmt='%s')
                np.savetxt(os.path.join(noisey_sn_dir, "type_{0}".format(sn_types[np.where(model_names == i)[0][0]]), '{0}_{1}_{2}_{3}.txt'.format(i, tau_idx, np.round(sn_a_v, 2), np.round(z_igx, 2))), noisey_sn_spectra, delimiter=',', fmt='%s')
                np.savetxt(os.path.join(convolved_spectra_dir, '{0}_{1}_{2}_{3}_{4}'.format(i, tau_idx, noisey_hg_file_names[val], np.round(sn_a_v, 2), np.round(z_igx, 2))), convolved_spectra, delimiter=',', fmt='%s')
                number_of_templates = number_of_templates + 1

print(number_of_templates)
#Turn into a dataframe for exporting
pure_sn_spectra_df = pd.DataFrame.from_dict(pure_sn_spectra_dict, orient = 'index')
noisey_sn_spectra_df = pd.DataFrame.from_dict(noisey_sn_spectra_dict, orient = 'index')
convolved_spectra_df = pd.DataFrame.from_dict(convolved_spectra_dict, orient = 'index')

#export the names of the spectra 
pure_sn_names = np.array(pure_sn_spectra_df.index.values)
noisey_sn_names = np.array(noisey_sn_spectra_df.index.values)
pure_hg_names = np.array(pure_hg_spectra_df.index.values)
noisey_hg_names = np.array(noisey_hg_spectra_df.index.values)
convolved_spectra_names = np.array(convolved_spectra_df.index.values)

np.savetxt(os.path.join(pure_sn_dir, 'metadata', 'pure_sn_spectra_names.txt'), pure_sn_names, delimiter=',', fmt='%s')
np.savetxt(os.path.join(noisey_sn_dir, 'metadata', 'noisey_sn_spectra_names.txt'), noisey_sn_names, delimiter=',', fmt='%s')
np.savetxt(os.path.join(pure_hg_dir, 'metadata', 'pure_hg_spectra_names.txt'), pure_hg_names, delimiter=',', fmt='%s')
np.savetxt(os.path.join(noisey_hg_dir, 'metadata', 'noisey_hg_spectra_names.txt'), noisey_hg_names, delimiter=',', fmt='%s')
np.savetxt(os.path.join(convolved_spectra_dir, 'metadata', 'convolved_spectra_names.txt'), convolved_spectra_names, delimiter=',', fmt='%s')

#Export the dataframes to a json file
noisey_sn_spectra_df.to_json(os.path.join(noisey_sn_dir,'metadata' ,'noisey_sn_spectra_data.json'))
pure_sn_spectra_df.to_json(os.path.join(pure_sn_dir, 'metadata','pure_sn_spectra_data.json'))
pure_hg_spectra_df.to_json(os.path.join(pure_hg_dir, 'metadata', 'pure_hg_spectra_data.json'))
noisey_hg_spectra_df.to_json(os.path.join(noisey_hg_dir,'metadata','noisey_hg_spectra_data.json'))
convolved_spectra_df.to_json(os.path.join(convolved_spectra_dir,'metadata','convolved_spectra_data.json'))


print('---------------------------')
print('  Training data generated  ')
print('---------------------------')