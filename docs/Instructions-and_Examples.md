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
| Redshift | `-z`, `--redshift` | `float` | Force SIREN to fit at a known redshift. | `None` |
| Batch Mode | `-b`, `--batch` | `boolean` | Run SIREN in single object or batch mode. | `false` (single object mode)|
| Metadata Path | `--metadata_path` | `string` | Path to the .csv file with any prior information for fitting.  | `None` |
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

### Examples
#### Single Object
Lets say we want to fit a single transient spectrum between the redshift range of (0.3 < z < 0.5) with a precision of 0.01. We also know that the spectrum has telluric features in that we dont want to fit and that we want to save the outputs to a specific folder. We can run the following:

```bash
#cd into the SIREN directory/make sure we're in the directory
cd SIREN_MACOS

#run SIREN!
./bin/SIREN -f path/to/your/spectrum.dat --z_min 0.3 --z_max 0.5 --z_step 0.01 --savepath path/to/your/desired/output/ -t true

```

#### Batch Mode
We may want to fit multiple spectra at once, we can set the batchmode flag to true and it will fit all these objects with the global defaults in the table above.
```bash
#cd into the SIREN directory/make sure we're in the directory
cd SIREN_MACOS

#run SIREN!
./bin/SIREN -f path/to/your/spectra/ -b true 
```
<h5>Fitting with a custom information for each transient</h5>
Now lets say we have 100 spectra, that we want customised fits for. Some may have host galaxy redshifts, some may have a rough redshift range given the survey limitations.  Some of the spectra have telluric features, some do not. Regardless we want the types. We can use a metadata.csv file to customise and reconcile all the individual inputs for each object. The metadata file should be structured like so:

| filename |	redshift|	z_min|	z_max	|z_step|	type |remove_telluric|	minimum_overlap	|	
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|Example1.txt	|	| 0.0|	0.5	|0.01	|	|N	|90	|N|
|Example2.txt	|0.32	| |		|	|	|Y	|90	|N|

We can then pass this csv to SIREN, which will read it in and fit with that information. 

```bash
#cd into the SIREN directory/make sure we're in the directory
cd SIREN_MACOS

#run SIREN!
./bin/SIREN -f path/to/your/spectra/ -b true --metadata_path path/to/your/metadata.csv
```

