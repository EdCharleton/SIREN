import os
import shutil
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def add_rows(row):
    """Decode Galaxy/SNe-type/Smag/Gmag/z/Texp from a Milligan25-style filename."""
    filename = str(row['filename'])
    smag = filename.find('Smag')
    gmag = filename.find('Gmag')
    z    = filename.find('z')
    texp = filename.find('texp')
    slash = filename.rfind('/')

    Gal = filename[slash + 1:smag - 3]
    SNe = filename[smag - 3: smag]
    if SNe[-1] == '_':
        SNe = SNe[:-1]
    Smag = filename[smag + 4: gmag]
    Gmag = filename[gmag + 4: z]
    Z    = filename[z + 1: texp]
    Texp = filename[texp + 4: -4]

    # Smag is cast to float here since the magnitude cut below needs it numeric
    # immediately; redshift/Texp are cast later (see get_redshift/get_texp)
    # to preserve full precision without an intermediate rounding step.
    return pd.Series([Gal, SNe, float(Smag), Gmag, Z, Texp],
                      index=['Galaxy', 'type', 'Fibre_Smag', 'Fibre_Gmag', 'redshift', 'Texp'])


def get_redshift(row):
    """Works for both spectra rows ('redshift') and obs rows ('redshift_estimate')."""
    return float(row.get('redshift', row.get('redshift_estimate')))


def get_texp(row):
    """Works for both spectra rows ('Texp') and obs rows ('visit_texp')."""
    return float(row.get('Texp', row.get('visit_texp')))


def extract_spec_info(spectra, output_file, obs_file,
                       dest_dir=None, mag_cut=22.5,
                       tol=0.001, require_obs_match=False,
                       save_obs_copy=None):
    """
    Full Milligan25 spectra pipeline:
      1. Decode Galaxy/type/Smag/Gmag/redshift/Texp from each filename.
      2. Apply the fibre-magnitude cut (Fibre_Smag < mag_cut).
      3. Match surviving spectra against the observation log (obs_file) on
         (redshift, Texp) via a KD-tree, within tolerance `tol`.
      4. Extract the phase from the TEMPLATE column into a separate 'phase' column.
      5. Optionally move the surviving (and, if require_obs_match=True,
         successfully-matched) spectra files into dest_dir.
      6. Write the combined metadata + obs-match CSV to output_file.
    """

    # ── 1. Parse filenames ────────────────────────────────────────────────────
    spectra = pd.DataFrame(list(spectra), columns=['filename'])
    spectra[['Galaxy', 'type', 'Fibre_Smag', 'Fibre_Gmag', 'redshift', 'Texp']] = \
        spectra.apply(add_rows, axis=1)

    # Keep the full path around for the move step later — 'filename' gets
    # overwritten to just the basename for the CSV output / matching.
    spectra['filepath'] = spectra['filename']
    spectra['filename'] = spectra['filename'].apply(os.path.basename)

    n_total = len(spectra)

    # ── 2. Magnitude cut ───────────────────────────────────────────────────────
    spectra = spectra[(spectra['Fibre_Smag'].values < mag_cut) ].reset_index(drop=True)
    n_after_cut = len(spectra)
    print(f"Magnitude cut (Fibre_Smag < {mag_cut}): {n_total} -> {n_after_cut} spectra")

    # ── 3. Match against the observation log via KD-tree ──────────────────────
    obs = pd.read_csv(obs_file)

    # Convert to pure floats without rounding, for tolerance-based matching
    spectra['z_float']    = spectra.apply(get_redshift, axis=1)
    spectra['Texp_float'] = spectra.apply(get_texp, axis=1)
    obs['z_float']        = obs.apply(get_redshift, axis=1)
    obs['Texp_float']     = obs.apply(get_texp, axis=1)

    obs_coords  = obs[['z_float', 'Texp_float']].values
    obs_tree    = cKDTree(obs_coords)
    spec_coords = spectra[['z_float', 'Texp_float']].values
    matches     = obs_tree.query_ball_point(spec_coords, r=tol, p=np.inf)

    obs_match_cols = ['TEMPLATE', 'name_x', 'u_obj_id', 'redshift_estimate', 'visit_texp']

    merged_rows = []

    for spec_idx, obs_indices in enumerate(matches):
        spec_row = spectra.iloc[spec_idx].to_dict()

        if obs_indices:
            # Multiple matches within tolerance — pick the closest one
            distances = [max(abs(spec_coords[spec_idx][0] - obs_coords[i][0]),
                              abs(spec_coords[spec_idx][1] - obs_coords[i][1]))
                         for i in obs_indices]
            best_obs_idx = obs_indices[np.argmin(distances)]
            obs_row = obs.iloc[best_obs_idx][obs_match_cols].to_dict()
        else:
            # No match within tolerance — pad with NaNs, keep the spectrum row
            obs_row = {col: np.nan for col in obs_match_cols}

        merged_rows.append({**spec_row, **obs_row})

    merged = pd.DataFrame(merged_rows)
    merged = merged.drop(columns=['z_float', 'Texp_float'])
    merged = merged.drop_duplicates(subset=['filename']).reset_index(drop=True)

    # ── 4. Extract Phase from TEMPLATE string ──────────────────────────────────
    # Matches 'phase' followed by optional negative sign (-), digits, and optional decimals
    merged['phase'] = merged['TEMPLATE'].str.extract(r'phase(-?\d+(?:\.\d+)?)')[0].astype(float)

    n_matched = merged['TEMPLATE'].notna().sum()
    print(f"Obs-log matches: {n_matched} / {len(merged)} spectra matched "
          f"(tolerance={tol}, p=inf)")

    merged_names   = np.array(merged['filename'].to_list(), dtype=str)
    original_names = np.array(spectra['filename'].to_list(), dtype=str)
    diff = np.setdiff1d(original_names, merged_names)
    if len(diff) > 0:
        print(f"WARNING: {len(diff)} spectra present after the mag cut but "
              f"missing from the merged output: {diff}")

    if require_obs_match:
        n_before_req = len(merged)
        merged = merged[merged['TEMPLATE'].notna()].reset_index(drop=True)
        print(f"require_obs_match=True: {n_before_req} -> {len(merged)} spectra "
              f"(dropped unmatched)")

    if save_obs_copy is not None:
        os.makedirs(os.path.dirname(save_obs_copy) or '.', exist_ok=True)
        obs.to_csv(save_obs_copy, index=False)
        print(f"Obs log copy saved to: {save_obs_copy}")

    # ── 5. Move the passing spectra into dest_dir ──────────────────────────────
    if dest_dir is not None:
        os.makedirs(dest_dir, exist_ok=True)

        moved = 0
        for src_path in merged['filepath']:
            dst_path = os.path.join(dest_dir, os.path.basename(src_path))
            try:
                shutil.move(src_path, dst_path)
                moved += 1
            except FileNotFoundError:
                print(f"  WARNING: source file not found, skipping: {src_path}")
            except Exception as e:
                print(f"  WARNING: could not move '{src_path}': {e}")

        print(f"Moved {moved} / {len(merged)} spectra to {dest_dir}")

    # Drop the helper column before writing the CSV
    merged = merged.drop(columns=['filepath'])

    merged.to_csv(output_file, index=False)
    print(f"Metadata written to: {output_file}")

    return merged


extract_spec_info(
    glob.glob('/Users/edcharleton/PhD/SIREN/spectra_to_classify/Milligan25_spectra/*.txt'),
    output_file='/Users/edcharleton/PhD/SIREN/spectra_to_classify/Milligan25_maglimited_info.csv',
    obs_file='/Users/edcharleton/PhD/SIREN/spectra_to_classify/Milligan25_observed_objects.csv',
    dest_dir='/Users/edcharleton/PhD/SIREN/spectra_to_classify/Milligan25_maglimited_spectra',
    mag_cut=22.5
)