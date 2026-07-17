#!/usr/bin/env python3
"""
Re-run the cell coincidence analysis for all NewCellAnalysis directories to generate:
  1. Updated percentile-0 cell CSV with a cell_id_in_image column
  2. oligomers_in_cells_percentile_0.csv  —  one row per in-cell oligomer with cell_id_in_image

Usage:
  python scripts/run_cell_analysis_for_db.py --test   # cingulate/microglia only
  python scripts/run_cell_analysis_for_db.py --full   # all 38 analysis directories

Outputs are written alongside the existing NewCellAnalysis CSV files.
Directories that already have oligomers_in_cells_percentile_0.csv are skipped.
"""

import argparse
import os
import re
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from AnalysisFunctions import Analysis_Functions

DATA_ROOT = (
    "/scratch/sycamore-asap/ASAP_Imaging_Data/Main_Survey/20241105_oligomers_in_cells"
)
ANALYSIS_SUBDIR = "NewCellAnalysis_OnlyLowerThreshold_NewPhotonThreshold"

# Analysis parameters matching the original runs
PHOTON_THRESHOLD = 150.0
LOWER_CELL_SIZE = 5000
UPPER_CELL_SIZE = 200000
SPACING = (0.5, 0.11, 0.11)
BLUR_DEGREE = 4
CELL_STRING = "C0"
PROTEIN_STRING = "C1"

P0_PATTERN = re.compile(
    r"single_cell_coincidence_.*_percentile_0_.*abovethreshold\.csv$"
)

ALL_TARGETS = [
    ("caudate", "microglia"),
    ("caudate", "neurons"),
    ("cingulate", "astrocytes"),
    ("cingulate", "microglia"),
    ("cingulate", "neurons"),
    ("frontal", "astrocytes"),
    ("frontal", "microglia"),
    ("frontal", "neurons"),
    ("parahippocampal", "microglia"),
    ("parahippocampal", "neurons"),
    ("parietal", "astrocytes"),
    ("parietal", "microglia"),
    ("parietal", "neurons"),
    ("putamen", "astrocytes"),
    ("putamen", "microglia"),
    ("putamen", "neurons"),
    ("substantiaNigra", "microglia"),
    ("temporal", "microglia"),
    ("temporal", "neurons"),
]


def process(af, region, cell_type, condition):
    analysis_dir = os.path.join(DATA_ROOT, region, cell_type, f"{condition}_analysis")
    spot_csv = os.path.join(analysis_dir, "spot_analysis.csv")
    new_cell_dir = os.path.join(analysis_dir, ANALYSIS_SUBDIR)
    oligo_out = os.path.join(new_cell_dir, "oligomers_in_cells_percentile_0.csv")

    if not os.path.isfile(spot_csv):
        print(f"  SKIP  no spot_analysis.csv: {analysis_dir}")
        return
    if not os.path.isdir(new_cell_dir):
        print(f"  SKIP  no NewCellAnalysis dir: {analysis_dir}")
        return
    if os.path.isfile(oligo_out):
        print(f"  SKIP  already complete: {oligo_out}")
        return

    # Find the percentile-0 cell CSV to overwrite
    p0_candidates = [f for f in os.listdir(new_cell_dir) if P0_PATTERN.match(f)]
    if not p0_candidates:
        print(f"  SKIP  no percentile-0 cell CSV in: {new_cell_dir}")
        return
    cell_csv_out = os.path.join(new_cell_dir, p0_candidates[0])

    print(f"  Loading spots from {spot_csv} ...")
    spot_data = (
        pl.scan_csv(spot_csv)
        .filter(pl.col("sum_intensity_in_photons") > PHOTON_THRESHOLD)
        .collect()
    )
    print(f"  {len(spot_data):,} spots above threshold")

    result = af.number_of_puncta_per_segmented_cell_with_threshold(
        analysis_file=spot_csv,
        analysis_data_raw=spot_data,
        threshold_lower=PHOTON_THRESHOLD,
        lower_cell_size_threshold=LOWER_CELL_SIZE,
        upper_cell_size_threshold=UPPER_CELL_SIZE,
        blur_degree=BLUR_DEGREE,
        cell_string=CELL_STRING,
        protein_string=PROTEIN_STRING,
        dims=3,
        spacing=SPACING,
        collect_oligomers=True,
    )

    if result is None or not isinstance(result, tuple):
        print(f"  WARNING no output for {region}/{cell_type}/{condition}")
        return

    cell_df, oligo_df = result

    if cell_df is not None and isinstance(cell_df, pl.DataFrame):
        cell_df.write_csv(cell_csv_out)
        print(f"  Saved {len(cell_df):,} cells → {cell_csv_out}")
    else:
        print(f"  WARNING no cell DataFrame returned")

    if oligo_df is not None and isinstance(oligo_df, pl.DataFrame):
        oligo_df.write_csv(oligo_out)
        print(f"  Saved {len(oligo_df):,} in-cell oligomers → {oligo_out}")
    else:
        print(f"  WARNING no in-cell oligomers found (check cell masks exist)")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run cell analysis to generate cell_id_in_image and oligomer linkage files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test", action="store_true", help="Run on cingulate/microglia only"
    )
    group.add_argument(
        "--full", action="store_true", help="Run on all 38 analysis directories"
    )
    args = parser.parse_args()

    af = Analysis_Functions()
    targets = [("cingulate", "microglia")] if args.test else ALL_TARGETS

    for region, cell_type in targets:
        for condition in ("HC", "PD"):
            print(f"\n=== {region} / {cell_type} / {condition} ===")
            process(af, region, cell_type, condition)

    print("\nDone.")


if __name__ == "__main__":
    main()
