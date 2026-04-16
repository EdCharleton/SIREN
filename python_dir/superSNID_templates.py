import os
import re
import shutil
import difflib
import numpy as np
import pandas as pd

metadata_path = "/Users/edcharleton/PhD/SIREN/Super-SNID Files/Super-SNID Spectra - Dylan's Finds.csv"
supersnid_dir_path = "/Users/edcharleton/PhD/SIREN/Super-SNID Files/Trimmed Spectra"
supersnid_spectra_path = "/Users/edcharleton/PhD/SIREN/sne_supersnid"

supersnid_metadata = pd.read_csv(metadata_path).dropna(subset=['Subtype'])

supersnid_template_names = supersnid_metadata['Template name'].astype(str).values
supersnid_template_types = supersnid_metadata['Subtype'].astype(str).values

all_files = os.listdir(supersnid_dir_path)

# Prepare output folders
for template_type in np.unique(supersnid_template_types):
    os.makedirs(os.path.join(supersnid_spectra_path, template_type), exist_ok=True)

# Helper to normalize (remove non-alnum)
def normalize(s):
    return re.sub(r'[^0-9a-z]', '', s.lower())

# Precompute normalized filenames for fallback fuzzy matching
norm_to_orig = { normalize(f): f for f in all_files }

#create a dictionary to hold the file names and redshift values
dict_metadata = {}



# Main loop
unmatched = []
for tname, subtype in zip(supersnid_template_names, supersnid_template_types):

    t = tname.strip()
    t_low = t.lower()

    # 1) direct substring match (case-insensitive)
    matches = [f for f in all_files if t_low in f.lower()]

    # 2) fallback: normalized difflib matching if no substring hits
    if not matches:
        norm_t = normalize(t)
        # find closest normalized filenames (keys of norm_to_orig)
        close_norms = difflib.get_close_matches(norm_t, norm_to_orig.keys(), n=3, cutoff=0.7)
        matches = [norm_to_orig[k] for k in close_norms]

    if matches:
        for matched_file in matches:
            src = os.path.join(supersnid_dir_path, matched_file)
            dst = os.path.join(supersnid_spectra_path, subtype, matched_file)
            try:
                shutil.copy2(src, dst)   # copy2 preserves metadata
                #print(f"Copied: {matched_file} -> {subtype}")
            except Exception as e:
                print(f"FAILED copying {matched_file}: {e}")
    else:
        unmatched.append(tname)
        print(f"No match found for template: {tname}")

    #Where there is a match, extract the redshift from the metadata file
    redshift_match = supersnid_metadata[supersnid_metadata['Template name'] == tname]['Redshift']
    
    #Add the redshift and file name to the dictionary
    for matched_file in matches:
        dict_metadata[matched_file] = redshift_match.values[0] if not redshift_match.empty else None

print(f"\nDone. {len(unmatched)} templates had no match (listed above).")
#Save the dictionary to a CSV file
metadata_df = pd.DataFrame(list(dict_metadata.items()), columns=['File Name', 'Redshift'])
metadata_df.to_csv(os.path.join(supersnid_spectra_path, 'supersnid_metadata.csv'), index=False)