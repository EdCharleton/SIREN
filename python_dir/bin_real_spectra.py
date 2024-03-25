import numpy as np
import spectres

four_most_wl = np.arange(3790, 9530, 20)
# Import a test real spectrum trimming the first 23 rows
convolved_spectra_file = np.loadtxt("real_spectra_dir/SN2011fe_2011-09-09_05-50-17_UH88_SNIFS_SNfactory.dat", skiprows=23)
convolved_spectra = convolved_spectra_file[:, 1]
convolved_spectra_wl = convolved_spectra_file[:, 0]

convolved_spectra = convolved_spectra / np.max(convolved_spectra) *100

convolved_spectra_trimmed = spectres.spectres(four_most_wl, convolved_spectra_wl, convolved_spectra)



# Save the trimmed spectrum 
np.savetxt("real_spectra_dir/2011fe.dat", np.column_stack((four_most_wl, convolved_spectra_trimmed)), fmt="%10.5f")