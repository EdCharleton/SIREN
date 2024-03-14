#Code to attribute the noisey spectrum to the clean spectrum

# Path: chi_squared_minimsation.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import os
import sncosmo
import scipy
import pandas as pd
from tqdm import tqdm
import shutil
from lmfit import Model
import warnings
import astropy.units as u
#-----------------------------------------------------

def delete_folder_contents(folder_path):
    """
    Function that deletes the contents of a folder
    Parameters:
        folder_path - path to the folder
    Returns:
        None
    """
    # Iterate over all items in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # Check if it's a file or a directory
        if os.path.isfile(item_path):
            # If it's a file, delete it
            os.remove(item_path)
        elif os.path.isdir(item_path):
            # If it's a directory, delete it recursively
            shutil.rmtree(item_path)

def adaptive_binning(altered_spectra, initial_spectra):
    """
    Function that bins the spectra if the number of counts within a region is less than 20 to put it in the gaussian regime
    Parameters:
        altered_spectra - spectra to be binned
        initial_spectra - spectra to be binned
    Returns:
        binned_altered_spectra - binned spectra
        binned_initial_spectra - binned spectra
    """
    binned_altered_spectra = np.array([])
    binned_initial_spectra = np.array([])

    i = 0
    while i < len(altered_spectra):
        if altered_spectra[i] < 20:
            if i == len(altered_spectra)-1:
                binned_altered_spectra = np.append(binned_altered_spectra, altered_spectra[i])
                binned_initial_spectra = np.append(binned_initial_spectra, initial_spectra[i])
                i += 1
            else:
                binned_altered_spectra = np.append(binned_altered_spectra, altered_spectra[i] + altered_spectra[i+1])
                binned_initial_spectra = np.append(binned_initial_spectra, initial_spectra[i] + initial_spectra[i+1])
                i += 2
        else:
            binned_altered_spectra = np.append(binned_altered_spectra, altered_spectra[i])
            binned_initial_spectra = np.append(binned_initial_spectra, initial_spectra[i])
            i += 1
    return binned_altered_spectra, binned_initial_spectra

def normalise_spectra(spectra, A):
    """
    Function that normalises a spectra
    Parameters:
        spectra - spectra to be normalised
        A - normalisation constant
    Returns:
        normalised spectra
    """
    return A*spectra

def chi_squared_classifier(initial_spectra_info, altered_spectra_info, show_plot =False, file_name_end = ''):
    """
    Function that classifies the altered spectra using a chi squared minimisation method
    Parameters: 
        initial_spectra_info - dataframe containing the information of the initial spectra
        altered_spectra_info - dataframe containing the information of the altered spectra
        show_plot - boolean to show or not to show the plot of the spectra
        file_name_end - string to add to the end of the file name
    Returns:
        closest_match - closest match to the altered spectra
        best_fit - index of the closest match
        identified - boolean to indicate whether the closest match is the same as the altered spectra
    
    """
    #generate empty arrays
    chi_squared_array = np.array([])
    initial_spectra_names = np.array([])
    normalisation_array = np.array([])

    #Number of types of supernovae and number of times to iterate through each type
    number_of_types = int(len(initial_spectra_info['Type'].unique()))
    iterations_per_type = int(initial_spectra_info.shape[0]/number_of_types)

    #Randomly select a spectra from the altered spectra
    random_index = np.random.randint(0, number_of_types*iterations_per_type)
    altered_spectra = np.loadtxt(os.path.join(altered_spectra_info['path'].iloc[random_index], altered_spectra_info.index[random_index]), delimiter=',')

    count = 0
    for i in range(number_of_types):
        for j in range(iterations_per_type):

            #import the initial spectra
            initial_spectra = np.loadtxt(os.path.join(initial_spectra_info['path'].iloc[count], initial_spectra_info.index[count]), delimiter=',')

            #Normalise the spectra
            normalisation_guess = np.max(altered_spectra)/np.max(initial_spectra)
            popt, pcov = scipy.optimize.curve_fit(normalise_spectra, initial_spectra, altered_spectra, p0=normalisation_guess)
            normalisation = popt[0]
            normalised_initial_spectra = initial_spectra*normalisation

            #Bin the two spectra if the number of counts within a region is less than 20 to put it in the gaussian regime
            adaptively_binned_altered_spectra, adaptively_binned_initial_spectra = adaptive_binning(altered_spectra, normalised_initial_spectra)

            #Low limit count poisson statistics
            #Eqn 10 from Gehrels 1986
            #sigma_u = altered_spectra + 1*np.sqrt(altered_spectra + 1) + (1**2 + 2)/3
            #Eqn 14 from Gehrels 1986
            #sigma_l = altered_spectra*(1-(1/(9*altered_spectra)) - 1/(3*np.sqrt(altered_spectra)))**3

            chi_squared = np.sum(np.where(adaptively_binned_altered_spectra != 0, (abs(adaptively_binned_initial_spectra - adaptively_binned_altered_spectra) ** 2) / adaptively_binned_altered_spectra, 0))
            
            chi_squared_array = np.append(chi_squared_array, chi_squared)
            normalisation_array = np.append(normalisation_array, normalisation)
            initial_spectra_names = np.append(initial_spectra_names, initial_spectra_info.index[count])

            count = count + 1

    best_fit = np.argmin(chi_squared_array)
    #print(length_of_adaptively_binned_spectra[best_fit], len(altered_spectra))

    if best_fit == random_index:
        #print('The noisey supernova was correctly identified')
        identified = True
    #Else plot the incorrect spectra and save it to a folder
    else:
        #print('The noisey supernova was incorrectly identified')
        identified = False
        closest_match =  np.loadtxt(os.path.join(initial_spectra_info['path'].iloc[best_fit], initial_spectra_info.index[best_fit]), delimiter=',')
        closest_match = closest_match*normalisation_array[best_fit]

        incorrect_match =  np.loadtxt(os.path.join(initial_spectra_info['path'].iloc[random_index], initial_spectra_info.index[random_index]), delimiter=',')
        incorrect_match = incorrect_match*normalisation_array[random_index]

        fig, ax = plt.subplots()
        rcParams["figure.figsize"] = [12, 8]
        rcParams["figure.dpi"] = 300
        rcParams["axes.linewidth"] = 3
        plt.rcParams.update({"font.size" : 13})
        plt.tick_params(axis = "both", which = "major", width = 1.5, length = 6, direction = "in")
        plt.tick_params(axis = "both", which = "minor", width = 1.5, length = 4, direction = "in")
        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel('Flux [Counts]')
        ax.xaxis.set_minor_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        plt.plot(four_most_wl, altered_spectra, color='black')
        plt.plot(four_most_wl, closest_match, color = 'red', label = r'$\chi^2 = {0}$'.format(np.round(chi_squared_array[best_fit], 2)))
        plt.plot(four_most_wl, incorrect_match, color = 'blue', label = r'$\chi^2 = {0}$'.format(np.round(chi_squared_array[random_index], 2)))
        plt.legend()
        plt.savefig(os.path.join('/Users/edcharleton/PhD/year_1/classification_engine/classification_practise/incorrect_matches', '{1}_{0}.png'.format(initial_spectra_info.index[best_fit], file_name_end)), bbox_inches='tight')
        plt.close()
    
    #If show plot is true, then save the plots to a folder called chi_squared_plots
    if show_plot == True:
        closest_match =  np.loadtxt(os.path.join(initial_spectra_info['path'].iloc[best_fit], initial_spectra_info.index[best_fit]), delimiter=',')
        closest_match = closest_match*normalisation_array[best_fit]
        fig, ax = plt.subplots()
        rcParams["figure.figsize"] = [12, 8]
        rcParams["figure.dpi"] = 300
        rcParams["axes.linewidth"] = 3
        plt.rcParams.update({"font.size" : 13})
        plt.tick_params(axis = "both", which = "major", width = 1.5, length = 6, direction = "in")
        plt.tick_params(axis = "both", which = "minor", width = 1.5, length = 4, direction = "in")
        ax.set_xlabel(r'Wavelength [$\AA$]]')
        ax.set_ylabel('Flux [Counts]')
        ax.xaxis.set_minor_locator(MultipleLocator(500))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        plt.plot(four_most_wl, altered_spectra, label='noisey spectra', color='black')
        plt.plot(four_most_wl, closest_match, label='clean spectra', color = 'red')
        plt.savefig(os.path.join('/Users/edcharleton/PhD/year_1/classification_engine/classification_practise/chi_squared_plots', '{1}_{0}.png'.format(initial_spectra_info.index[best_fit], file_name_end)), bbox_inches='tight')
        plt.close()

    return initial_spectra_info.index[best_fit], best_fit, identified
    

#-----------------------------------------------------
#Main section of code - run the chi squared classifier 
#-----------------------------------------------------

#Import the data
four_most_wl = np.arange(3700, 9500, 10)
clean_spectra_info = pd.read_json(os.path.join('spectra_files','clean_spectra_data.json'))
noisey_spectra_info = pd.read_json(os.path.join('spectra_files','noisey_spectra_data.json'))

#Delete the previous contents of the plot directories for a fresh run
delete_folder_contents('/Users/edcharleton/PhD/year_1/classification_engine/classification_practise/incorrect_matches')
delete_folder_contents('/Users/edcharleton/PhD/year_1/classification_engine/classification_practise/chi_squared_plots')

#Supress any warnings that may occur
warnings.filterwarnings("ignore")

no_of_iterations = 100
success_rate_count = 0

print('-'*52)
print('Running the chi squared classifier for {0} iterations'.format(no_of_iterations))
print('-'*52)


for i in tqdm(range(no_of_iterations)): 
    name_of_best_match, best_fit_index, success = chi_squared_classifier(clean_spectra_info, noisey_spectra_info, show_plot = True, file_name_end=i)
    if success == True:
        success_rate_count = success_rate_count + 1

success_rate = (success_rate_count/no_of_iterations) * 100
print('-'*52)
print('The chi squared classifier was successful for {0} out of {1} iterations ({2}%)'.format(success_rate_count, no_of_iterations, np.round(success_rate, 2)))
print('-'*52)








