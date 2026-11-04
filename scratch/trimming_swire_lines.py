import numpy as np
import os
from matplotlib import pyplot as plt
import spectres

trimmed_path = "host_galaxy_gen_dir/swire_library/trimmed"
to_trim_path = "host_galaxy_gen_dir/swire_library/swire_backups"
file_names = os.listdir(to_trim_path)
#remove .DS_Store 
file_names = [f for f in file_names if f != '.DS_Store']

# Load the data
for name in file_names:
    template_wavelength, template_flux = np.loadtxt(os.path.join(to_trim_path, name), unpack=True)

    SIREN_range = np.where((template_wavelength > 2000) & (template_wavelength < 12000))
    template_wavelength = template_wavelength[SIREN_range]
    template_flux = template_flux[SIREN_range]
    #rebin the templates

    new_range = np.arange(2000, 12000, 15)
    new_template_flux = spectres.spectres(new_range, template_wavelength, template_flux)

    plt.title(name)
    plt.plot(new_range, new_template_flux, color = 'red', alpha = 0.2)
    plt.scatter(template_wavelength, template_flux, alpha = 1, color = "black", s = 10) 
    plt.plot(template_wavelength, template_flux, color = "black", alpha = 0.5)
    plt.show()





