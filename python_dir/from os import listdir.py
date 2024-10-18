from os import listdir
from os.path import isfile, join
from astropy.io import fits



folder_path = "/Users/edcharleton/PhD/SIREN/grace_help/fits_files"

file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

for name in file_names:
    #read in your fits file
    hdul = fits.open(folder_path + "/" +name)
    print(hdul.info())