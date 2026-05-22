import numpy as np
from matplotlib import pyplot as plt
import spectres

spec_name = "/Users/edcharleton/PhD/SIREN/spectra_to_classify/Dwayne/spectra/SN2011fe_2011-08-25_00-00-00_TNG_DOLORES_PTF.ascii"
spec_name = "/Users/edcharleton/PhD/SIREN/spectra_to_classify/Dwayne/spectra/TiDES-SN26aan.ascii"
try:
    wavs, spec, spec_err = np.loadtxt(spec_name, unpack=True)
except ValueError:
    wavs, spec = np.loadtxt(spec_name, unpack=True)

# Normalize the spectrum
spec = spec *1e16
print(spec)
# Bin up the spectrum
resolution = np.arange(3500, 9000, 15)
spec_high_res = spectres.spectres(resolution, wavs, spec)


n = int(len(resolution) / 2)

newest_wavs = resolution[::2][:n]

gradient = np.zeros(n)
for i in range(1, n - 1):
    j = i * 2  
    gradient[i] = (spec_high_res[j + 2] - spec_high_res[j - 2]) / (resolution[j + 2] - resolution[j - 2])


second_derivative = np.zeros(n)
for i in range(1, n - 1):
    second_derivative[i] = (gradient[i + 1] - gradient[i - 1]) / (newest_wavs[i + 1] - newest_wavs[i - 1])

plt.figure(figsize=(10, 9))

plt.subplot(2, 1, 1)
plt.plot(wavs, spec, label='Spectrum')
plt.plot(resolution, spec_high_res, label='Binned Spectrum', color='red')
plt.ylim(0,2)
plt.title('SN2011fe Spectrum')
plt.xlabel('Wavelength (Å)')
plt.xlim(3500, 8000)
plt.ylabel('Flux')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(newest_wavs, abs(gradient)3, label='First Derivative', color='orange')
plt.title('First Derivative of the Spectrum')
plt.xlabel('Wavelength (Å)')
plt.ylabel('dF/dλ')
plt.xlim(3500, 8000)
plt.legend()

"""
plt.subplot(3, 1, 3)
plt.plot(newest_wavs, second_derivative, label='Second Derivative', color='green')
plt.title('Second Derivative of the Spectrum')
plt.xlabel('Wavelength (Å)')
plt.ylabel('d^2F/dλ^2')
plt.xlim(3500, 8000)
plt.legend()
"""

plt.tight_layout()
plt.show()