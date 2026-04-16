import numpy as np

spectrum = np.loadtxt('/Users/edcharleton/PhD/SIREN/SIREN_test_sets/Dwayne/spectra/SN2011fe_2011-08-25_00-00-00_TNG_DOLORES_PTF.ascii', dtype=float)

wl, spectrum = spectrum[:,0], spectrum[:,1]

content = np.column_stack((wl, spectrum))

np.savetxt('/Users/edcharleton/PhD/SIREN/SIREN_test_sets/Dwayne/spectra/SN2011fe.txt', content, fmt = "%1.5f", delimiter = ",")