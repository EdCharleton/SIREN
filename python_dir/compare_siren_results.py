#!/usr/bin/env python3
"""
compare_siren_results.py

Unit-test SIREN's classification output ("Dwayne") against known ground-truth
types and redshifts from the metadata CSV.

Produces:
  - dwayne_merged_results.csv      full row-by-row comparison table
  - dwayne_confusion_fine.png      confusion matrix, full SIREN type list
  - dwayne_confusion_coarse.png    confusion matrix, supernova super-type
  - dwayne_redshift_comparison.png true vs. fitted redshift scatter + residuals
  - dwayne_classification_report.csv   per-type precision/recall/F1 (fine-grained)
  - Console summary of accuracy, top-5 accuracy, redshift agreement, and any
    filenames that §ld not be matched between the two files.

Usage:
    python3 compare_siren_results.py \
        --csv dwayne_metadata.csv \
        --json dwayne_unit_tes_classifications.json \
        --outdir .
"""

import argparse
import json
import re
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ═══════════════════════════════════════════════════════════════════════════
# 1. SIREN's canonical type vocabulary
# ═══════════════════════════════════════════════════════════════════════════
# Flattened from the list supplied — used to sanity-check that every mapped
# "true" type actually exists in SIREN's template library. If it doesn't,
# SIREN can never get that classification right by construction, which is
# an important distinction from SIREN simply guessing wrong.
SIREN_CANONICAL_TYPES = {
    "20xnd", "computed", "FBOT", "Ia", "Ia-02cx", "Ia-02es", "Ia-91bg",
    "Ia-91t", "Ia-99aa", "Ia-Ca-rich", "Ia-CSM", "Ia-CSM-(ambigious)",
    "Ia-pec", "Ia-rapid", "Ia-SC", "Ib", "Ib_c", "Ib-Ca-rich", "Ib-norm",
    "Ib-pec", "Ibn", "Ic", "Ic-BL", "Ic-broad", "Ic-Ca-rich", "Ic-norm",
    "Ic-pec", "Icn", "II", "II-flash", "II-pec", "IIb", "IIb-flash", "IIL",
    "IIn", "IIn-pec", "IIP", "ILRT", "LBV", "LRN", "metadata", "SLSN-I",
    "SLSN-Ib", "SLSN-II", "SLSN-IIb", "SLSN-IIn", "SN-Imposter",
    "super-chandra", "TDE", "TDE-Featless", "TDE-H", "TDE-H-He", "TDE-He",
}

# "computed" and "metadata" look like stray non-type artifacts (e.g. leftover
# folder/column names) rather than real SN classes — flagged, not removed.
SUSPICIOUS_CANONICAL_ENTRIES = {"computed", "metadata"}

# ═══════════════════════════════════════════════════════════════════════════
# 2. Free-text → SIREN canonical type mapping
# ═══════════════════════════════════════════════════════════════════════════
# The metadata CSV uses looser/older naming conventions than SIREN's template
# library. Add new entries here as new mismatches are discovered — anything
# NOT in this dict is passed through unchanged and checked against
# SIREN_CANONICAL_TYPES below.
TYPE_ALIASES = {
    "Ia-91T-like":        "Ia-91t",
    "SN 91-bg-like":      "Ia-91bg",
    "Ia-91bg-like":       "Ia-91bg",
    "SN-Ia-91bg-like":    "Ia-91bg",
    "Ib/c":               "Ib_c",
}


def normalise_type(raw_type: str) -> str:
    """Apply the alias table; pass through unchanged otherwise."""
    raw_type = str(raw_type).strip()
    return TYPE_ALIASES.get(raw_type, raw_type)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Coarse "super-type" grouping (for a more legible confusion matrix)
# ═══════════════════════════════════════════════════════════════════════════
# 66 real spectra spread across ~30 SIREN sub-types produces an extremely
# sparse fine-grained confusion matrix. This coarse grouping trades
# granularity for readability. Prefix-based, with the exceptions below.
SUPERTYPE_OVERRIDES = {
    "Unknown": "Unknown",
}


def supertype_of(fine_type: str) -> str:
    if fine_type in SUPERTYPE_OVERRIDES:
        return SUPERTYPE_OVERRIDES[fine_type]
    if (fine_type.startswith("Ia")) | (fine_type == "super-chandra"):
        return "Ia"
    if fine_type.startswith("Ib") or fine_type.startswith("Ic"):
        return "Ib/Ic"
    if fine_type.startswith("II"):
        return "II"
    if fine_type.startswith("SLSN"):
        return "SLSN"
    if fine_type.startswith("TDE"):
        return "TDE"
    return "Other/Exotic"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Filename matching between the CSV and the JSON
# ═══════════════════════════════════════════════════════════════════════════
def extract_token(stem: str) -> str:
    """
    Pull out the object-name token used for fuzzy matching: everything
    before the first underscore, with a leading 'SN' stripped ONLY when
    immediately followed by a digit (so 'SN2018gea' -> '2018gea', but
    survey-prefixed names like 'SNLS-06D3cn' are left alone).
    """
    token = stem.split("_")[0]
    token = re.sub(r"^SN(?=\d)", "", token, flags=re.IGNORECASE)
    return token.lower()


def match_filenames(csv_stems, json_stems):
    """
    Three-tier matching:
      1. Exact stem match
      2. Object-token match (handles 'SN' prefix / suffix differences)
      3. Fuzzy match (difflib) as a last resort, reported for manual review

    Returns: dict csv_stem -> json_stem (or None), plus a list of
             (csv_stem, tier, note) diagnostics for anything not exact.
    """
    json_stems = list(json_stems)
    json_set = set(json_stems)
    json_token_map = {}
    for js in json_stems:
        tok = extract_token(js)
        json_token_map.setdefault(tok, []).append(js)

    matches = {}
    diagnostics = []

    for cs in csv_stems:
        # Tier 1 — exact
        if cs in json_set:
            matches[cs] = cs
            continue

        # Tier 2 — token match
        tok = extract_token(cs)
        candidates = json_token_map.get(tok, [])
        if len(candidates) == 1:
            matches[cs] = candidates[0]
            diagnostics.append((cs, "token", f"matched to '{candidates[0]}'"))
            continue
        elif len(candidates) > 1:
            matches[cs] = None
            diagnostics.append((cs, "AMBIGUOUS", f"token '{tok}' matched {len(candidates)} JSON entries: {candidates}"))
            continue

        # Tier 3 — fuzzy
        close = get_close_matches(cs, json_stems, n=1, cutoff=0.6)
        if close:
            matches[cs] = close[0]
            diagnostics.append((cs, "fuzzy", f"matched to '{close[0]}' (verify manually)"))
        else:
            matches[cs] = None
            diagnostics.append((cs, "UNMATCHED", "no corresponding entry found in SIREN output"))

    return matches, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════
def main(csv_path, json_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ───────────────────────────────────────────────────────────
    df_meta = pd.read_csv(csv_path)

    df_meta = df_meta[df_meta["Fibre_Gmag"] < 21.5]


    dupes = df_meta[df_meta.duplicated("filename", keep=False)]
    if not dupes.empty:
        print(f"Note: {dupes['filename'].nunique()} duplicate filename(s) in metadata CSV — keeping first occurrence.")
        for fn in dupes["filename"].unique():
            print(f"  duplicated: {fn}")
    df_meta = df_meta.drop_duplicates("filename", keep="first").reset_index(drop=True)

    df_meta["stem"] = df_meta["filename"].apply(lambda f: Path(str(f)).stem)
    df_meta["true_type_raw"] = df_meta["type_ig"]
    df_meta["true_type"] = df_meta["type_ig"].apply(normalise_type)
    df_meta["true_z"] = pd.to_numeric(df_meta["redshift"], errors="coerce")

    # Sanity-check every mapped true type against SIREN's real vocabulary
    unknown_true_types = sorted(set(df_meta["true_type"]) - SIREN_CANONICAL_TYPES)
    if unknown_true_types:
        print("WARNING: the following ground-truth types (after alias mapping) do not")
        print("appear in SIREN's canonical type list — SIREN cannot possibly predict")
        print("these correctly, since no matching template exists. Add an alias in")
        print("TYPE_ALIASES, or confirm SIREN's template library covers this type:")
        for t in unknown_true_types:
            print(f"  - {t!r}")
        print()

    # ── Load JSON ──────────────────────────────────────────────────────────
    with open(json_path) as f:
        results = json.load(f)

    json_stems = list(results.keys())

    # ── Match filenames ───────────────────────────────────────────────────
    matches, diagnostics = match_filenames(df_meta["stem"].tolist(), json_stems)

    unmatched = [cs for cs, note in [(d[0], d) for d in diagnostics] if matches[cs] is None]
    fuzzy_or_token = [d for d in diagnostics if d[1] in ("token", "fuzzy")]
    ambiguous = [d for d in diagnostics if d[1] == "AMBIGUOUS"]

    print(f"Filename matching: {len(df_meta) - len(unmatched)} / {len(df_meta)} matched "
          f"({sum(1 for d in diagnostics if d[1]=='fuzzy')} fuzzy, "
          f"{sum(1 for d in diagnostics if d[1]=='token')} token-based).")

    if fuzzy_or_token:
        print("\nNon-exact matches (recommend spot-checking these):")
        for cs, tier, note in fuzzy_or_token:
            print(f"  [{tier:>6}] {cs}  ->  {note}")

    if ambiguous:
        print("\nAMBIGUOUS matches (skipped — please resolve manually):")
        for cs, tier, note in ambiguous:
            print(f"  {cs}: {note}")

    if unmatched:
        print(f"\n{len(unmatched)} metadata entries have NO corresponding SIREN output "
              f"(never classified, or classification failed):")
        for cs in unmatched:
            print(f"  - {cs}")

    # ── Build merged comparison table ─────────────────────────────────────
    rows = []
    for _, row in df_meta.iterrows():
        js_key = matches.get(row["stem"])
        if js_key is None or js_key not in results:
            continue
        r = results[js_key]
        top_fits = r.get("top_fits", [])

        # Unique transient types in rank order (dedup, preserve order) for top-N accuracy
        ranked_types = []
        for fit in top_fits:
            t = fit.get("transient_type")
            if t not in ranked_types:
                ranked_types.append(t)

        true_t = row["true_type"]
        rank_of_true = ranked_types.index(true_t) + 1 if true_t in ranked_types else None

        rows.append({
            "filename":        row["filename"],
            "json_key":        js_key,
            "true_type":       true_t,
            "true_type_raw":   row["true_type_raw"],
            "pred_type":       r.get("type", "Unknown"),
            "true_supertype":  supertype_of(true_t),
            "pred_supertype":  supertype_of(r.get("type", "Unknown")),
            "true_z":          row["true_z"],
            "pred_z":          r.get("redshift"),
            "chi_sq_red":      r.get("chi_sq_red"),
            "correct_fine":    true_t == r.get("type", "Unknown"),
            "correct_coarse":  supertype_of(true_t) == supertype_of(r.get("type", "Unknown")),
            "rank_of_true_type": rank_of_true,   # None if not in top-N at all
            "survey":          row.get("Survey"),
        })

    merged = pd.DataFrame(rows)
    merged["dz"] = merged["pred_z"] - merged["true_z"]
    merged["abs_dz"] = merged["dz"].abs()

    merged.to_csv(outdir / "dwayne_merged_results.csv", index=False)
    n = len(merged)
    print(f"\n{'='*60}\nCompared {n} spectra with matched ground truth.\n{'='*60}")

    # ── Accuracy summary ───────────────────────────────────────────────────
    acc_fine   = merged["correct_fine"].mean()
    acc_coarse = merged["correct_coarse"].mean()

    in_top5  = merged["rank_of_true_type"].apply(lambda r: r is not None and r <= 5).mean()
    in_top10 = merged["rank_of_true_type"].apply(lambda r: r is not None and r <= 10).mean()
    never_in_list = merged["rank_of_true_type"].isna().mean()

    print(f"\nTop-1 accuracy (fine-grained type):   {acc_fine:.1%}  ({merged['correct_fine'].sum()}/{n})")
    print(f"Top-1 accuracy (super-type):          {acc_coarse:.1%}  ({merged['correct_coarse'].sum()}/{n})")
    print(f"True type present in top-5 fits:      {in_top5:.1%}")
    print(f"True type present in top-10 fits:     {in_top10:.1%}")
    print(f"True type absent from ALL top-N fits: {never_in_list:.1%}  "
          f"(template coverage gap, or minimum_overlap/z-range excluded it)")

    # ── Redshift agreement ─────────────────────────────────────────────────
    valid_z = merged.dropna(subset=["true_z", "pred_z"])
    if len(valid_z) > 0:
        med_abs_dz = valid_z["abs_dz"].median()
        frac_001 = (valid_z["abs_dz"] < 0.01).mean()
        frac_005 = (valid_z["abs_dz"] < 0.05).mean()
        print(f"\nRedshift: median |Δz| = {med_abs_dz:.4f}")
        print(f"          |Δz| < 0.01: {frac_001:.1%}   |Δz| < 0.05: {frac_005:.1%}")

    # ── Per-type classification report (fine-grained) ─────────────────────
    report = classification_report(
        merged["true_type"], merged["pred_type"],
        zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(outdir / "dwayne_classification_report.csv")
    print(f"\nPer-type precision/recall/F1 saved to dwayne_classification_report.csv")

    # ── Confusion matrix — fine-grained ────────────────────────────────────
    fine_labels = sorted(set(merged["true_type"]) | set(merged["pred_type"]))
    cm_fine = confusion_matrix(merged["true_type"], merged["pred_type"], labels=fine_labels)

    fig, ax = plt.subplots(figsize=(max(8, len(fine_labels) * 0.5), max(6, len(fine_labels) * 0.45)))
    sns.heatmap(cm_fine, annot=True, fmt="d", cmap="Blues",
                xticklabels=fine_labels, yticklabels=fine_labels,
                cbar_kws={"label": "Count"}, ax=ax)
    ax.set_xlabel("SIREN predicted type")
    ax.set_ylabel("True type")
    ax.set_title(f"SIREN classification — fine-grained type\n(n={n}, top-1 accuracy={acc_fine:.1%})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(outdir / "dwayne_confusion_fine.png", dpi=200)
    plt.close(fig)

    # ── Confusion matrix — coarse super-type ───────────────────────────────
    coarse_labels = sorted(set(merged["true_supertype"]) | set(merged["pred_supertype"]))
    cm_coarse = confusion_matrix(merged["true_supertype"], merged["pred_supertype"], labels=coarse_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_coarse, annot=True, fmt="d", cmap="Blues",
                xticklabels=coarse_labels, yticklabels=coarse_labels,
                cbar_kws={"label": "Count"}, ax=ax)
    ax.set_xlabel("SIREN predicted super-type")
    ax.set_ylabel("True super-type")
    ax.set_title(f"SIREN classification — super-type\n(n={n}, top-1 accuracy={acc_coarse:.1%})")
    plt.tight_layout()
    fig.savefig(outdir / "dwayne_confusion_coarse.png", dpi=200)
    plt.close(fig)

    # ── Redshift comparison plot ────────────────────────────────────────────
    if len(valid_z) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = valid_z["correct_fine"].map({True: "tab:green", False: "tab:red"})
        ax1.scatter(valid_z["true_z"], valid_z["pred_z"], c=colors, alpha=0.7, edgecolor="k", linewidth=0.3)
        lims = [0, max(valid_z["true_z"].max(), valid_z["pred_z"].max()) * 1.05]
        ax1.plot(lims, lims, "k--", linewidth=1, label="y = x")
        ax1.set_xlabel("True redshift")
        ax1.set_ylabel("SIREN redshift")
        ax1.set_title("Redshift recovery\n(green = correct type, red = wrong type)")
        ax1.legend()
        ax1.set_xlim(lims); ax1.set_ylim(lims)

        ax2.hist(valid_z["dz"], bins=30, color="steelblue", edgecolor="k")
        ax2.axvline(0, color="k", linestyle="--", linewidth=1)
        ax2.set_xlabel("Δz  (SIREN − true)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Redshift residuals\nmedian |Δz| = {med_abs_dz:.4f}")

        plt.tight_layout()
        fig.savefig(outdir / "dwayne_redshift_comparison.png", dpi=200)
        plt.close(fig)

    print(f"\nAll outputs written to: {outdir.resolve()}")
    print("  dwayne_merged_results.csv        — full per-object comparison table")
    print("  dwayne_confusion_fine.png        — fine-grained confusion matrix")
    print("  dwayne_confusion_coarse.png      — super-type confusion matrix")
    print("  dwayne_redshift_comparison.png   — redshift scatter + residuals")
    print("  dwayne_classification_report.csv — per-type precision/recall/F1")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="dwayne_metadata.csv")
    parser.add_argument("--json", default="dwayne_unit_tes_classifications.json")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    main(args.csv, args.json, args.outdir)
