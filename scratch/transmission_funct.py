import numpy as np
from matplotlib import pyplot as plt
import spectres

file = np.loadtxt("src/transmission_function.txt")
wl = file[:,0]
transmission = file[:,1]


transmission_indices = np.where(transmission < -0.0005)
transmission_mask = wl[transmission_indices]
transmission_function = transmission[transmission_indices]

tellurics = np.array([])
tellurics = np.append(tellurics, transmission_mask[0])


for i in range(len(transmission_mask)-1):
    
    if transmission_mask[i+1] - transmission_mask[i] > 10:
        tellurics = np.append(tellurics, transmission_mask[i])
        tellurics = np.append(tellurics, transmission_mask[i+1])
tellurics = np.append(tellurics, transmission_mask[-1])

print(tellurics)

np.savetxt("src/metadata/transmission_function_resampled.txt", np.column_stack((wl, transmission)), fmt = "%1.5f")
np.savetxt("src/metadata/transmission_mask.txt", transmission_mask, fmt = "%1.5f")
np.savetxt("src/metadata/tellurics.txt", tellurics, fmt = "%1.5f")