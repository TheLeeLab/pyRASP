#!/usr/bin/env python3
"""
Build main_survey_database_v2.duckdb from analysed imaging data.

Usage:
  python scripts/build_database.py --test   # cingulate/microglia only
  python scripts/build_database.py --full   # all data

Run run_cell_analysis_for_db.py first to generate oligomers_in_cells_percentile_0.csv
and the updated cell CSVs with cell_id_in_image before running this script.
"""

import argparse
import os
import re
import sys

import duckdb
import polars as pl
import docx

DATA_ROOT = "/scratch/sycamore-asap/ASAP_Imaging_Data/Main_Survey/20241105_oligomers_in_cells"
SI_DOCX = os.path.expanduser("~/Downloads/SI.docx")
DB_PATH = "/scratch/duckdb-database/main_survey_database_v2.duckdb"
ANALYSIS_SUBDIR = "NewCellAnalysis_OnlyLowerThreshold_NewPhotonThreshold"

P0_PATTERN = re.compile(
    r"single_cell_coincidence_.*_percentile_0_.*abovethreshold\.csv$"
)
RE_MINCELL = re.compile(r"mincellsize_(\d+)")
RE_MAXCELL = re.compile(r"maxcellsize_(\d+)")
RE_PATIENT = re.compile(r"/(HC\d+|PD\d+)/")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(path: str) -> str:
    return path.replace("\\", "/")


def patient_from_path(path: str):
    m = RE_PATIENT.search(norm(path))
    return m.group(1) if m else None


def int_or_none(v: str):
    return int(v) if v not in ("NA", "", "N/A") else None


def float_or_none(v: str):
    return float(v) if v not in ("NA", "", "N/A") else None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id  VARCHAR PRIMARY KEY,
    condition   VARCHAR,
    sex         VARCHAR,
    age_of_onset        INTEGER,
    age_at_death        INTEGER,
    disease_duration    INTEGER,
    pmi_hours           DOUBLE,
    npd                 VARCHAR,
    asyn_braak          INTEGER,
    tau_braak           INTEGER,
    abeta_thal          INTEGER
);

CREATE TABLE IF NOT EXISTS brain_regions (
    region_id   INTEGER PRIMARY KEY,
    patient_id  VARCHAR REFERENCES patients(patient_id),
    brain_region VARCHAR,
    condition   VARCHAR
);

CREATE TABLE IF NOT EXISTS cell_types (
    cell_type_id INTEGER PRIMARY KEY,
    region_id    INTEGER REFERENCES brain_regions(region_id),
    patient_id   VARCHAR,
    brain_region VARCHAR,
    cell_type    VARCHAR
);

CREATE TABLE IF NOT EXISTS cells (
    cell_id             INTEGER PRIMARY KEY,
    cell_id_in_image    INTEGER,
    cell_type_id        INTEGER REFERENCES cell_types(cell_type_id),
    patient_id          VARCHAR REFERENCES patients(patient_id),
    brain_region        VARCHAR,
    cell_type           VARCHAR,
    min_cell_size_vox   INTEGER,
    max_cell_size_vox   INTEGER,
    area_um3            DOUBLE,
    x_centre            DOUBLE,
    y_centre            DOUBLE,
    z_centre            DOUBLE,
    puncta_cell_likelihood  DOUBLE,
    n_puncta_in_cell        INTEGER,
    oligomer_concentration_per_um3  DOUBLE,
    image_filename      VARCHAR
);

CREATE TABLE IF NOT EXISTS oligomers (
    oligomer_id         BIGINT PRIMARY KEY,
    cell_id             INTEGER REFERENCES cells(cell_id),
    cell_type_id        INTEGER REFERENCES cell_types(cell_type_id),
    patient_id          VARCHAR REFERENCES patients(patient_id),
    brain_region        VARCHAR,
    cell_type           VARCHAR,
    in_cell             BOOLEAN,
    x                   DOUBLE,
    y                   DOUBLE,
    z                   DOUBLE,
    sum_intensity_in_photons    DOUBLE,
    bg_per_punctum      DOUBLE,
    bg_per_pixel        DOUBLE,
    zi                  DOUBLE,
    zf                  DOUBLE,
    image_filename      VARCHAR
);
"""


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def load_patients(conn):
    doc = docx.Document(SI_DOCX)
    table = doc.tables[0]
    rows = []
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        pid, sex, onset, death, duration, pmi, npd, asyn, tau, abeta = cells
        rows.append((
            pid,
            "HC" if pid.startswith("HC") else "PD",
            sex,
            int_or_none(onset),
            int_or_none(death),
            int_or_none(duration),
            float_or_none(pmi),
            npd,
            int_or_none(asyn),
            int_or_none(tau),
            int_or_none(abeta),
        ))
    conn.executemany(
        "INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    print(f"  {len(rows)} patients loaded")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="cingulate/microglia only")
    group.add_argument("--full", action="store_true", help="all data")
    args = parser.parse_args()

    targets = [("cingulate", "microglia")] if args.test else ALL_TARGETS

    conn = duckdb.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()

    # ---- patients ----
    print("\n[1] Patients")
    load_patients(conn)

    # Surrogate key counters (resume from existing data if re-run)
    region_id_ctr = conn.execute("SELECT COALESCE(MAX(region_id),0) FROM brain_regions").fetchone()[0]
    ct_id_ctr = conn.execute("SELECT COALESCE(MAX(cell_type_id),0) FROM cell_types").fetchone()[0]
    cell_id_ctr = conn.execute("SELECT COALESCE(MAX(cell_id),0) FROM cells").fetchone()[0]
    oligo_id_ctr = conn.execute("SELECT COALESCE(MAX(oligomer_id),0) FROM oligomers").fetchone()[0]

    # Lookup dicts (in-memory; small)
    region_map = {}   # (patient_id, brain_region) -> region_id
    ct_map = {}       # (region_id, cell_type)     -> cell_type_id
    # (norm_image_filename, cell_id_in_image) -> global cell_id  —  built during cells step
    img_cell_map = {}

    # Pre-load existing maps if resuming a partial run
    for row in conn.execute("SELECT patient_id, brain_region, region_id FROM brain_regions").fetchall():
        region_map[(row[0], row[1])] = row[2]
    for row in conn.execute("SELECT region_id, cell_type, cell_type_id FROM cell_types").fetchall():
        ct_map[(row[0], row[1])] = row[2]
    for row in conn.execute("SELECT image_filename, cell_id_in_image, cell_id FROM cells").fetchall():
        img_cell_map[(row[0], row[1])] = row[2]

    # ---- cells ----
    print("\n[2] Cells")
    for region, cell_type in targets:
        for condition in ("HC", "PD"):
            new_cell_dir = os.path.join(
                DATA_ROOT, region, cell_type, f"{condition}_analysis", ANALYSIS_SUBDIR
            )
            if not os.path.isdir(new_cell_dir):
                continue
            p0_files = [f for f in os.listdir(new_cell_dir) if P0_PATTERN.match(f)]
            if not p0_files:
                continue
            cell_csv = os.path.join(new_cell_dir, p0_files[0])

            min_vox = int(RE_MINCELL.search(p0_files[0]).group(1)) if RE_MINCELL.search(p0_files[0]) else None
            max_vox = int(RE_MAXCELL.search(p0_files[0]).group(1)) if RE_MAXCELL.search(p0_files[0]) else None

            try:
                cdf = pl.read_csv(cell_csv)
            except Exception as e:
                print(f"  ERROR reading {cell_csv}: {e}")
                continue

            if "cell_id_in_image" not in cdf.columns:
                print(f"  WARNING missing cell_id_in_image in {cell_csv} — run run_cell_analysis_for_db.py first")
                continue

            rows_cells = []
            for row in cdf.to_dicts():
                img_fn = norm(row["image_filename"])
                pid = patient_from_path(img_fn)
                if pid is None:
                    continue

                rkey = (pid, region)
                if rkey not in region_map:
                    region_id_ctr += 1
                    region_map[rkey] = region_id_ctr
                    cond_str = "HC" if pid.startswith("HC") else "PD"
                    conn.execute(
                        "INSERT OR IGNORE INTO brain_regions VALUES (?,?,?,?)",
                        [region_id_ctr, pid, region, cond_str],
                    )

                ctkey = (region_map[rkey], cell_type)
                if ctkey not in ct_map:
                    ct_id_ctr += 1
                    ct_map[ctkey] = ct_id_ctr
                    conn.execute(
                        "INSERT OR IGNORE INTO cell_types VALUES (?,?,?,?,?)",
                        [ct_id_ctr, region_map[rkey], pid, region, cell_type],
                    )

                area = row.get("area/um3")
                conc = (row["n_puncta_in_cell"] / area) if area and area > 0 else None
                cid_in_img = int(row["cell_id_in_image"])

                # Skip if already in DB (resume safety)
                if (img_fn, cid_in_img) in img_cell_map:
                    continue

                cell_id_ctr += 1
                img_cell_map[(img_fn, cid_in_img)] = cell_id_ctr
                rows_cells.append((
                    cell_id_ctr, cid_in_img, ct_map[ctkey], pid, region, cell_type,
                    min_vox, max_vox, area,
                    row.get("x_centre"), row.get("y_centre"), row.get("z_centre"),
                    row.get("puncta_cell_likelihood"),
                    int(row["n_puncta_in_cell"]) if row["n_puncta_in_cell"] is not None else None,
                    conc, img_fn,
                ))

            if rows_cells:
                conn.executemany("INSERT INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows_cells)
                conn.commit()
                print(f"  {region}/{cell_type}/{condition}: {len(rows_cells):,} cells")

    # ---- in-cell oligomers ----
    print("\n[3] In-cell oligomers")
    for region, cell_type in targets:
        for condition in ("HC", "PD"):
            oligo_csv = os.path.join(
                DATA_ROOT, region, cell_type, f"{condition}_analysis",
                ANALYSIS_SUBDIR, "oligomers_in_cells_percentile_0.csv"
            )
            if not os.path.isfile(oligo_csv):
                print(f"  SKIP (missing): {oligo_csv}")
                continue

            # Skip if already loaded (resume safety)
            n_existing_ic = conn.execute(
                "SELECT COUNT(*) FROM oligomers WHERE in_cell=TRUE AND brain_region=? AND cell_type=? AND patient_id LIKE ?",
                [region, cell_type, condition + "%"]
            ).fetchone()[0]
            if n_existing_ic > 0:
                print(f"  SKIP already loaded {n_existing_ic:,} in-cell for {region}/{cell_type}/{condition}")
                continue

            try:
                odf = pl.read_csv(oligo_csv)
            except Exception as e:
                print(f"  ERROR reading {oligo_csv}: {e}")
                continue

            batch = []
            skipped = 0
            for row in odf.to_dicts():
                img_fn = norm(row["image_filename"])
                pid = patient_from_path(img_fn)
                if pid is None:
                    skipped += 1
                    continue
                cid_in_img = int(row["cell_id_in_image"])
                global_cell_id = img_cell_map.get((img_fn, cid_in_img))
                ctid = ct_map.get((region_map.get((pid, region)), cell_type))

                oligo_id_ctr += 1
                batch.append((
                    oligo_id_ctr, global_cell_id, ctid, pid, region, cell_type,
                    True,
                    row.get("x"), row.get("y"), row.get("z"),
                    row.get("sum_intensity_in_photons"),
                    row.get("bg_per_punctum"), row.get("bg_per_pixel"),
                    row.get("zi"), row.get("zf"), img_fn,
                ))
                if len(batch) >= 50_000:
                    conn.executemany("INSERT INTO oligomers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                    conn.commit()
                    batch = []

            if batch:
                conn.executemany("INSERT INTO oligomers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                conn.commit()
            inserted = conn.execute("SELECT COUNT(*) FROM oligomers WHERE in_cell=TRUE").fetchone()[0]
            print(f"  {region}/{cell_type}/{condition}: done (skipped {skipped} rows with no patient ID)")

    # ---- out-of-cell oligomers (via DuckDB native CSV reading) ----
    print("\n[4] Out-of-cell oligomers")
    for region, cell_type in targets:
        for condition in ("HC", "PD"):
            spot_csv = os.path.join(
                DATA_ROOT, region, cell_type, f"{condition}_analysis", "spot_analysis.csv"
            )
            if not os.path.isfile(spot_csv):
                continue

            # Skip if already loaded (resume safety)
            n_existing = conn.execute(
                "SELECT COUNT(*) FROM oligomers WHERE in_cell=FALSE AND brain_region=? AND cell_type=? AND patient_id LIKE ?",
                [region, cell_type, condition + "%"]
            ).fetchone()[0]
            if n_existing > 0:
                print(f"  SKIP already loaded {n_existing:,} out-of-cell for {region}/{cell_type}/{condition}")
                continue

            # Detect whether this file has an incell column (some older files don't)
            with open(spot_csv) as _f:
                _header = _f.readline().strip().split(',')
            has_incell = 'incell' in _header

            # We use DuckDB's native CSV reader for streaming — avoids loading 6 GB into RAM.
            print(f"  {region}/{cell_type}/{condition}: streaming {spot_csv} (incell={'yes' if has_incell else 'no'}) ...")

            # Build a temp table of (patient_id, cell_type_id) pairs we know about
            known_pairs = [
                (pid, ctid)
                for (pid, reg), rid in region_map.items()
                for (rid2, ct), ctid in ct_map.items()
                if rid == rid2 and reg == region and ct == cell_type
            ]
            if not known_pairs:
                print(f"    SKIP: no cell_type entries found for this combination")
                continue

            current_max = conn.execute("SELECT COALESCE(MAX(oligomer_id),0) FROM oligomers").fetchone()[0]

            conn.execute("CREATE OR REPLACE TEMP TABLE pid_ct (patient_id VARCHAR, cell_type_id INTEGER)")
            conn.executemany("INSERT INTO pid_ct VALUES (?,?)", known_pairs)

            if has_incell:
                cols_spec = "{'y':'DOUBLE','x':'DOUBLE','z':'DOUBLE','sum_intensity_in_photons':'DOUBLE','bg_per_punctum':'DOUBLE','bg_per_pixel':'DOUBLE','incell':'VARCHAR','zi':'DOUBLE','zf':'DOUBLE','image_filename':'VARCHAR'}"
                where_clause = "WHERE TRY_CAST(s.incell AS DOUBLE) = 0.0"
                anti_join = ""
            else:
                # No incell column: anti-join against oligomers_in_cells_percentile_0.csv
                # to avoid double-counting spots already loaded as in_cell=TRUE.
                oligo_in_cell_csv = os.path.join(
                    DATA_ROOT, region, cell_type, f"{condition}_analysis",
                    ANALYSIS_SUBDIR, "oligomers_in_cells_percentile_0.csv"
                )
                cols_spec = "{'y':'DOUBLE','x':'DOUBLE','z':'DOUBLE','sum_intensity_in_photons':'DOUBLE','bg_per_punctum':'DOUBLE','bg_per_pixel':'DOUBLE','zi':'DOUBLE','zf':'DOUBLE','image_filename':'VARCHAR'}"
                where_clause = ""
                if os.path.isfile(oligo_in_cell_csv):
                    conn.execute(f"""
                        CREATE OR REPLACE TEMP TABLE incell_spots AS
                        SELECT
                            REPLACE(image_filename, '\\', '/') AS image_filename,
                            x, y, z
                        FROM read_csv('{oligo_in_cell_csv}',
                                      columns={{'x':'DOUBLE','y':'DOUBLE','z':'DOUBLE',
                                                'sum_intensity_in_photons':'DOUBLE',
                                                'bg_per_punctum':'DOUBLE','bg_per_pixel':'DOUBLE',
                                                'zi':'DOUBLE','zf':'DOUBLE',
                                                'image_filename':'VARCHAR','cell_id_in_image':'INTEGER'}})
                    """)
                    anti_join = """LEFT JOIN incell_spots ic
                        ON REPLACE(s.image_filename,'\\','/') = ic.image_filename
                        AND s.x = ic.x AND s.y = ic.y AND s.z = ic.z
                    WHERE ic.image_filename IS NULL"""
                else:
                    anti_join = ""

            conn.execute(f"""
                INSERT INTO oligomers
                SELECT
                    {current_max} + ROW_NUMBER() OVER ()          AS oligomer_id,
                    NULL                                           AS cell_id,
                    pc.cell_type_id                               AS cell_type_id,
                    REGEXP_EXTRACT(
                        REPLACE(s.image_filename, '\\', '/'),
                        '/(HC\\d+|PD\\d+)/', 1)                  AS patient_id,
                    '{region}'                                     AS brain_region,
                    '{cell_type}'                                  AS cell_type,
                    FALSE                                          AS in_cell,
                    s.x, s.y, s.z,
                    s.sum_intensity_in_photons,
                    s.bg_per_punctum,
                    s.bg_per_pixel,
                    s.zi, s.zf,
                    REPLACE(s.image_filename, '\\', '/')          AS image_filename
                FROM read_csv('{spot_csv}', columns = {cols_spec}) s
                JOIN pid_ct pc ON pc.patient_id = REGEXP_EXTRACT(
                        REPLACE(s.image_filename, '\\', '/'), '/(HC\\d+|PD\\d+)/', 1)
                {anti_join}
                {where_clause}
            """)
            conn.commit()
            n_oc = conn.execute("SELECT COUNT(*) FROM oligomers WHERE in_cell=FALSE AND brain_region=? AND cell_type=?",
                                [region, cell_type]).fetchone()[0]
            print(f"    running total out-of-cell for {region}/{cell_type}: {n_oc:,}")

    conn.execute("DROP TABLE IF EXISTS pid_ct")

    # ---- indexes ----
    print("\n[5] Creating indexes ...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cells_patient ON cells(patient_id, brain_region, cell_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_oligo_cell   ON oligomers(cell_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_oligo_patient ON oligomers(patient_id, brain_region, cell_type, in_cell)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_oligo_intensity ON oligomers(sum_intensity_in_photons)")
    conn.commit()

    # ---- summary ----
    print("\n=== Database summary ===")
    for tbl in ("patients", "brain_regions", "cell_types", "cells", "oligomers"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:25s}: {n:>12,}")
    ic = conn.execute("SELECT COUNT(*) FROM oligomers WHERE in_cell=TRUE").fetchone()[0]
    oc = conn.execute("SELECT COUNT(*) FROM oligomers WHERE in_cell=FALSE").fetchone()[0]
    print(f"    oligomers in-cell    : {ic:>12,}")
    print(f"    oligomers out-of-cell: {oc:>12,}")
    print(f"\nDatabase written to: {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
