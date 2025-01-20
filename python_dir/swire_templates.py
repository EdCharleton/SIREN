import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import spectres
from matplotlib import rcParams

swire_templates = os.listdir('host_galaxy_gen_dir/swire_library/masked')
#remove the readme file
#swire_templates.remove('Readme')

print(swire_templates)
"""
for template_name in swire_templates:
    if template_name == ".DS_Store":
        continue

    template = np.loadtxt('host_galaxy_gen_dir/swire_library/to_be_masked/' + template_name)
    template_wavelength = template[:,0]
    template_flux = template[:,1]

    SIREN_range = np.where((template_wavelength > 2000) & (template_wavelength < 12000))
    template_wavelength = template_wavelength[SIREN_range]
    template_flux = template_flux[SIREN_range]

    plt.title("SWIRE Template {0}".format(template_name))
    #plt.plot(template_wavelength, template_flux, color = "black")

    #rebin the templates

    new_range = np.arange(2000, 12000, 15)
    new_template_flux = spectres.spectres(new_range, template_wavelength, template_flux)

    #create a new column for the mask
    mask = np.ones(len(new_range))

#Plentiful O3 and Ha Hb lines in this array:     emission_region = np.array([[2660, 2890],[3720, 3750], [4750, 5050], [6400, 6800]])
#emission_region = np.array([[4890, 5050],  [9450, 9570]])

    emission_region = np.array([[4850, 5175]])
    #emission_region = emission_region.reshape(-1, 2)

    if emission_region.size == 0:
        mask = np.ones(len(new_range))
    else:
        for i in range(len(emission_region)):
            emission_line_locations = np.where((new_range > emission_region[i][0]) & (new_range < emission_region[i][1]))
            mask[emission_line_locations] = 0

    masked_data = new_template_flux[np.where(mask == 1)]
    masked_axis = new_range[np.where(mask == 1)]

    plt.scatter(masked_axis, masked_data, color = 'red')
    plt.plot(new_range, new_template_flux, color = 'black')
    plt.show()

    #save the masked template
    masked_template = np.column_stack((new_range, new_template_flux, mask))
    np.savetxt('host_galaxy_gen_dir/swire_library/masked/' + template_name, masked_template)
    break
"""

template_wl = np.arange(2315, 9015, 15)
for template_name in swire_templates:
    if template_name == ".DS_Store":
        continue

    #rebin the templates to the range of the SIREN templates
    template = np.loadtxt('host_galaxy_gen_dir/swire_library/masked/' + template_name)
    template_wavelength = template[:,0]
    template_flux = template[:,1]
    template_mask = template[:,2]

    new_template_flux = spectres.spectres(template_wl, template_wavelength, template_flux)

    #Adjust the mask to the new range
    template_min = np.min(template_wl)
    template_max = np.max(template_wl)

    mask_indices = np.where((template_wavelength >= template_min) & (template_wavelength <= template_max))

    new_template_mask = template_mask[mask_indices]

    masked_axis = template_wl[np.where(new_template_mask == 1)]

    plt.figure(figsize=(12, 8), dpi=300)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    rcParams.update({'font.size': 20})

    plt.title("{0}".format(template_name))
    plt.plot(template_wavelength, template_flux, color = "black", alpha = 0.5)
    plt.scatter(masked_axis, new_template_flux[np.where(new_template_mask == 1)], color = 'red')
    plt.plot(template_wl, new_template_flux, color = 'black')
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel("Flux")
    plt.savefig('host_galaxy_gen_dir/swire_library/plots/' + template_name + '.png')
    plt.close()

    #save the masked template
    masked_template = np.column_stack((template_wl, new_template_flux, new_template_mask))
    np.savetxt('host_galaxy_gen_dir/swire_library/swire_templates/' + template_name, masked_template)






