import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = "SIREN_test_sets/DES/DES_ALL_SPECTRA/DES13C1feu_AAT_2013_10_30.dat"

data = pd.read_csv(file_path, delimiter='\t', header = 15)

numpy_data = np.loadtxt(file_path,  skiprows=16)

print(numpy_data[1][0])
