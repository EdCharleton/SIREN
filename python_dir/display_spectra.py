import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

file_path = "/Users/edcharleton/PhD/SIREN/transient_template_dir_full/II/2002hh_2002hh_2003-04-08_00-00-00_Lick-3m_KAST_None.ascii_0.0.txt"
wavs, flux = np.loadtxt(file_path, unpack=True, delimiter=",")

plt.figure(figsize=(16,9), dpi = 200)
rcParams["axes.facecolor"] = "white"
params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)


ax = plt.gca()

#Set axes line widths to 1.5
ax.spines["top"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)

#Set tick widths to 1.5 and inner
ax.tick_params(width=1.5, direction="in", length=10)

#Add minor ticks
ax.minorticks_on()
ax.tick_params(which="minor", width=1.5, direction="in", length=5)

plt.plot(wavs, flux, color="#EEBA0B", label="1987A Spectrum at Peak Brightness")
plt.xlabel(r"Wavelength ($\AA$)", fontsize=22, color = "white")
plt.ylabel(r"Flux", fontsize=22, color = "white")

plt.xlim(3500, 8000)
#plt.ylim(-100, 1400)

#increase tick label size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig("/Users/edcharleton/PhD/spectrum.png", transparent=True, bbox_inches="tight")
plt.close()
