# Instructions & Examples

### Basic Usage

SIREN runs out of the command line, for mac and linux users, this will mean opening a terminal and cding into the SIREN directory. 

From there, we can now run SIREN. To run SIREN, the only thing we need is a path to the spectrum we want to classify. Siren primarily deals with plain text files (txt/ascii/dat etc.) and FITS files. SIREN should be able to handle the default arrangement for spectral files: Column 1 wavelength, column 2 flux, column 3 flux uncertainty (although it can run without uncertainties).


There are two running modes for SIREN, batch mode (more than one spectrum at a time) and single object. 



#### Single Object mode 
 ```bash
./bin/SIREN -f path/to/your/spectrum.ascii 
 ```

#### Batch mode 
```bash
./bin/SIREN -f path/to/the/folder/of/spectra/ -b true 
 ```
SIREN will export plots to a directory, as well as a JSON file with the classification, redshift and reduced chi2. SIREN tells you where the json classifications are saved to when it finishes running. By default, it saves in the classifications directory within your SIREN folder. 

### Further Fitting Customisation
SIREN can take various arguments to help customise the fits further. The current arguments SIREN can take are listed below:

| Option | SIREN Flag | Type | Explanation | Default |
| :--- | :--- | :--- | :--- | :--- |
| Redshift | `-z` | `float` | Force SIREN to fit at a known redshift. | `None` |
| Batch Mode | `-b`, `--batch` | `boolean` | Run SIREN in single object or batch mode. | `false` (single object mode)|
| Remove Tellurics | `-t`, `--remove_telluric` | `boolean` | Mask the parts of the spectrum where a telluric is present. | `false` |
| Galactic Source | `-g`, `--galactic_source` | `boolean` | **[Currently not working]** If the object is in our own Milky Way, fit at z=0. | `false` |
| Show Plot | `-d`, `--display_plot` | `boolean` | Use the interactive plotter. *(Not recommended in batch mode)*. | `true` in single object mode `false` in batch mode |
| Minimum Overlap | `-m`, `--minimum_overlap` | `float` (%) | Force the amount of overlap between templates and the spectrum (e.g., `90`). | `75` |
| Run on GPU | `--use_gpu` | `boolean` | Run SIREN on Metal (macOS) or CUDA (Linux) cores. | `false` |
| Save Path | `-s`, `--savepath` | `string` | Directory where the plot & JSON will be saved. | `classifications/` in `SIREN_APP` |
| Minimum Redshift | `--z_min` | `float` | Minimum redshift you want to fit. | `0.0` |
| Maximum Redshift | `--z_max` | `float` | Maximum redshift you want to fit. | `0.8` |
| Redshift Step | `--z_step` | `float` | Redshift precision to fit with. | `0.01` |
| Version | `--version` | `boolean` | Display the SIREN version and exit| `false` |

*Please note, all booleans need to be lowercase to match Julia syntax*