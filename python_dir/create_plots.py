import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


survey_name = 'DES'
metadata_path = os.path.join('SIREN_test_sets/{0}/{0}_metadata.csv'.format(survey_name)) 

#load metadata
survey_metadata = pd.read_csv(metadata_path)

#phases = pd.to_numeric(survey_metadata['Phase (days)'].values, errors='coerce')
redshifts = pd.to_numeric(survey_metadata['Redshift'].values, errors='coerce')
all_types = survey_metadata['Obj. Type'].values


condition = np.where((all_types != "SNIa"))[0]
condition_name = 'Non Ias'

file_names = survey_metadata['Ascii file'].values[condition]
redshifts = survey_metadata['Redshift'].values[condition]
types = all_types[condition]
#3phase_vals = phases[condition]


#print(survey_metadata.columns)
plot_save_path = os.path.join('SIREN_test_sets/{0}/{1}/'.format(survey_name, condition_name)) 



##import the spectra from the file_names variable
spectra_path = os.path.join('SIREN_test_sets/{0}/{0}_ALL_SPECTRA/'.format(survey_name))

#set the plot DPI to 300
#delete the folder if it already exists

for name in tqdm.tqdm(file_names):
    try:
        spectrum = np.loadtxt(spectra_path + name, skiprows=18)
        redshift = redshifts[np.where(file_names == name)]
        type = types[np.where(file_names == name)]

        plt.figure(dpi=150)
        plt.title("{0} \n Redshift: {1} - Type: {2} Phase (days):".format(name, redshift, type))
        plt.plot(spectrum[:,0], spectrum[:,1])
        plt.savefig('SIREN_test_sets/{0}/{2}/{1}.png'.format(survey_name, name, condition_name))
        plt.close()
    except:
        continue


