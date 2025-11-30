# %%
import pandas as pd
import re
import glob


# -------------------------------------------------
# Helper: extract block between two markers
# -------------------------------------------------
def extract_block(text, start, end):
    pattern = rf"{start}(.*?){end}"
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


# -------------------------------------------------
# Parse FITTING STATISTICS
# -------------------------------------------------
def parse_fitting_statistics(block):
    d = {}
    if not block:
        return d

    d["R_square"] = float(re.search(r"R SQUARE\s+([0-9.]+)", block).group(1))
    d["percent_mass"] = float(re.search(r"% MASS\s+([0-9.]+)", block).group(1))
    d["chi_square"] = float(re.search(r"CHI SQUARE\s+([0-9.]+)", block).group(1))
    dof = re.search(r"DEGREES FREEDOM\s+([0-9.]+)", block)
    d["degrees_freedom"] = float(dof.group(1)) if dof else None

    # FIT MEASURE is optional
    fm = re.search(r"FIT MEASURE\s+([0-9.]+)", block)
    d["fit_measure"] = float(fm.group(1)) if fm else None

    return d


# -------------------------------------------------
# Parse SOURCE CONTRIBUTION ESTIMATES
# -------------------------------------------------
def parse_source_contribution(block):
    rows = []
    if not block:
        return rows

    for line in block.splitlines():
        line = line.strip()
        if re.match(r"^(YES|NO)\s+", line):
            parts = line.split()
            # YES BBPDDY BIOPADDY 0.59853 0.07679 7.79454
            yes_no, code, name, sce, se, tstat = parts[:6]
            rows.append(
                {
                    "EST_CODE": code,
                    "NAME": name,
                    "SCE": float(sce),
                    "StdErr": float(se),
                    "Tstat": float(tstat),
                }
            )

    return rows


# -------------------------------------------------
# Parse SOURCE NAME TABLE
# -------------------------------------------------
def parse_source_name(block):
    if not block:
        return []

    # Keep raw lines but we'll strip when needed
    lines = block.splitlines()

    # ---- 1. Find header row: the one that starts with SPECIES (ignoring spaces) ----
    header_cols = None
    for l in lines:
        s = l.strip()
        if s.startswith("SPECIES"):
            header_cols = s.split()
            break

    if not header_cols:
        return []

    # Example: ['SPECIES','CALCULATED','MEASURED','BIOPAD','TRANSP']
    source_cols = header_cols[3:]  # dynamic number of sources

    rows = []

    # ---- 2. Parse data rows ----
    for l in lines:
        s = l.strip()
        if not s:
            continue
        # skip header and title line
        if s.startswith("SPECIES") or "SOURCE NAME" in s:
            continue

        # data lines: start with species code (TMAC, FLUC, PHEC, ...)
        if re.match(r"^[A-Za-z0-9]{3,4}\s", s):
            parts = s.split()
            if len(parts) < 3:
                continue

            species = parts[0]
            calculated = float(parts[1])
            measured = float(parts[2])
            contrib_vals = parts[3:]

            row = {
                "SPECIES": species,
                "CALCULATED": calculated,
                "MEASURED": measured,
            }

            # assign each value to the corresponding source name
            for src, val in zip(source_cols, contrib_vals):
                row[src] = float(val)

            rows.append(row)

    return rows


# -------------------------------------------------
# MAIN PARSER FOR ALL FILES
# -------------------------------------------------
def parse_cmb_folder(path_pattern):
    stats_list = []
    contrib_list = []
    sourcename_list = []

    for fname in glob.glob(path_pattern):
        id = fname.split("/")[-1].split("\\")[-1].replace(".txt", "")
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # ==== BLOCK EXTRACTION ====
        block_stats = extract_block(
            text, r"FITTING STATISTICS:", r"SOURCE CONTRIBUTION ESTIMATES:"
        )

        block_contrib = extract_block(
            text, r"SOURCE CONTRIBUTION ESTIMATES:", r"SPECIES CONCENTRATIONS:"
        )

        # SOURCE NAME comes AFTER SPECIES CONCENTRATIONS
        block_sourcename = extract_block(
            text, r"SOURCE NAME", r"$"  # until end of file
        )

        # ==== PARSE ====
        stats = parse_fitting_statistics(block_stats)
        stats["id"] = id

        contrib = parse_source_contribution(block_contrib)
        for c in contrib:
            c["id"] = id
        sourcename = parse_source_name(block_sourcename)
        for s in sourcename:
            s["id"] = id

        # ==== APPEND ====
        stats_list.append(stats)
        contrib_list.extend(contrib)
        sourcename_list.extend(sourcename)

    areas_dict = {"QN": "Quang Ninh", "HP": "Hai Phong", "HY": "Hung Yen"}
    station_type = {"RS": "Residential", "TR": "Traffic", "BG": "Background"}
    df_stats = pd.DataFrame(stats_list)
    df_stats["Province"] = df_stats["id"].apply(
        lambda x: areas_dict.get(x.split("_")[-1][:2], "Unknown")
    )
    df_stats["SiteType"] = df_stats["id"].apply(
        lambda x: station_type.get(x.split("_")[-1][2:], "Unknown")
    )
    df_stats.drop(columns=["id"], inplace=True)
    cols_stats = df_stats.columns.tolist()
    df_stats = df_stats[cols_stats[-2:] + cols_stats[:-2]]

    source_types = {
        "CRD": "Coal (Residential)",
        "VGS": "Gasoline vehicles",
        "VDI": "Diesel vehicles",
        "BBN": "Biomass burning",
        "NONRDDIE": "Non-road diesel engine",
        "TRANSPET": "Gasoline vehicles",
        "WASTE": "Waste burning",
    }

    df_contrib = pd.DataFrame(contrib_list)
    df_contrib["Source"] = df_contrib["NAME"].map(source_types)
    df_contrib["Province"] = df_contrib["id"].apply(
        lambda x: areas_dict.get(x.split("_")[-1][:2], "Unknown")
    )
    df_contrib["SiteType"] = df_contrib["id"].apply(
        lambda x: station_type.get(x.split("_")[-1][2:], "Unknown")
    )
    df_contrib.rename(columns={"SCE": "Source contribution estimates"}, inplace=True)
    df_contrib.drop(columns=["EST_CODE", "NAME", "id"], inplace=True)
    cols_contrib = df_contrib.columns.tolist()
    df_contrib = df_contrib[cols_contrib[-2:] + ["Source"] + cols_contrib[:-3]]

    df_source = pd.DataFrame(sourcename_list)
    return (
        df_stats,
        df_contrib,
        df_source,
    )


pah_path_ind = "./cmb_result/pah/india_sp/*.txt"
pah_path_seoul = "./cmb_result/pah/seoul_sp/*.txt"
df_stats_pah_ind, df_contrib_pah_ind, df_sourcename_pah_ind = parse_cmb_folder(
    pah_path_ind
)
df_stats_pah_seoul, df_contrib_pah_seoul, df_sourcename_pah_seoul = parse_cmb_folder(
    pah_path_seoul
)

btex_path = "./cmb_result/btex/*.txt"
df_stats_btex, df_contrib_btex, df_sourcename_btex = parse_cmb_folder(btex_path)

# %%
