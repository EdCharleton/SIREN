import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import json
from collections import OrderedDict
import extinction

print('---------------------------')
print('Generating extinction...   ')
print('---------------------------')

# Load spectra metadata
spectra_info = json.load(open(os.path.join("convolved_spectra_dir", "metadata", "convolved_spectra_data.json")), object_pairs_hook=OrderedDict)
#Save the JSON file to a directory in the dusty metadata folder
with open(os.path.join("dusty_spectra_dir", "metadata", "dusty_spectra_data.json"), "w") as json_file:
    json.dump(spectra_info, json_file, indent=4)

spectra_file_names = np.loadtxt(os.path.join("convolved_spectra_dir", "metadata", "convolved_spectra_names.txt"), dtype=str)
four_most_wl = np.arange(3790.0, 9530.0, 20.0)

# Loop over spectra
for spectra_key in spectra_file_names:
    spectra_path = spectra_info["path"][spectra_key]
    spectra_hg_contamination = spectra_info["host galaxy contamination"][spectra_key]
    file_path = os.path.join(spectra_path, str(spectra_key))
    spectra = np.loadtxt(file_path)

    #Randomly select an extinction between -0.2 and 1 mag
    a_v = np.random.uniform(-0.2, 1)
    r_v = 3.1

    #random_law = np.random.choice(["ccm89", "od94", "fp99"])
    dust_law = "ccm89"
    dusty_spectra = extinction.apply(extinction.ccm89(four_most_wl, a_v=a_v, r_v=r_v), spectra)

    dusty_file_path = os.path.join("dusty_spectra_dir", str(spectra_key) + "_dusty.txt")
    # Save the spectra
    np.savetxt(dusty_file_path, dusty_spectra)

    # Update the metadata
    
    spectra_info.setdefault("A_V", {})[spectra_key] = a_v
    spectra_info.setdefault("R_V", {})[spectra_key] = r_v
    spectra_info.setdefault("Dust Law", {})[spectra_key] = dust_law

# Update JSON file with new metadata
with open(os.path.join("dusty_spectra_dir", "metadata", "dusty_spectra_data.json"), "w") as json_file:
    json.dump(spectra_info, json_file, indent=4)

# Save the names of the spectra
np.savetxt(os.path.join("dusty_spectra_dir", "metadata", "dusty_spectra_names.txt"), spectra_file_names, fmt="%s")


print('---------------------------')
print('   Extinction generated!   ')
print('---------------------------')