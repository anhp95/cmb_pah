# %%
import argparse
import re
from pathlib import Path
from typing import Iterable
import numpy as np

import pandas as pd

NOT_USED_SPECIES = []

AD_SPECIES = [
    "Nap",
    "Acy",
    "Ace",
    "Flu",
    "Phe",
    "Ant",
    "Fla",
    "Pyr",
    "Chr",
    "BaA",
    "BbF",
    "BkF",
    "BaP",
    "Ind",
    "DbA",
    "BPer",
]


def prep_pah_ad_util(df, version):

    used_species = [s for s in AD_SPECIES if s not in NOT_USED_SPECIES]

    df["TMA"] = df[used_species].sum(axis=1)

    used_species = ["TMA"] + used_species

    species_c = {s: f"{s.upper()}C" for s in used_species}
    species_u = {s: f"{s.upper()}U" for s in used_species}

    keys = ["ID", "DATE", "DUR", "STHOUR", "SIZE"]

    mean_df = (
        df.groupby(["ID", "SIZE"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns=species_c)
    )

    std_df = (
        df.groupby(["ID", "SIZE"], as_index=False)
        .sem(numeric_only=True)
        .rename(columns=species_u)
    )
    result = pd.merge(mean_df, std_df, on=["ID", "SIZE"])
    result["DATE"] = ["01/07/25"] * len(result)
    result["DUR"] = [24] * len(result)
    result["STHOUR"] = [0] * len(result)
    result = result[
        keys + [f"{s.upper()}{n}" for s in used_species for n in ["C", "U"]]
    ].round(4)
    result_fine = result[result["SIZE"] == "FINE"]
    result_coarse = result[result["SIZE"] == "COARSE"]
    result_fine.to_csv(f"data/run/AD_{version}_fine.csv", index=False)
    result_coarse.to_csv(f"data/run/AD_{version}_coarse.csv", index=False)
    result.to_csv(f"data/run/AD_{version}.csv", index=False)

    return result


def prep_ad_v3(version="v3_all", ad_file_path="data/ambient/AD_merge_ng_v3.csv"):
    df = pd.read_csv(ad_file_path)

    rename_site = {
        "Background": "BG",
        "Traffic": "TR",
        "Residential": "RS",
    }
    df["Sitetype"] = df["Sitetype"].apply(lambda x: rename_site[x])
    df["ID"] = df["ID"].apply(lambda x: re.sub(r"\d+", "", x).strip())
    df["ID"] = df["ID"] + df["Sitetype"]
    df["SIZE"] = df["PM"].apply(lambda x: "COARSE" if "10" in x.strip() else "FINE")

    df = prep_pah_ad_util(df, version)
    return df


def prep_ad_v2(ad_file_path="data/ambient/AD_merge_micg.xlsx"):

    used_species = [s for s in AD_SPECIES if s not in NOT_USED_SPECIES]

    df = pd.read_excel(ad_file_path, sheet_name="Sheet1")

    df["ID"] = df["Station"].apply(
        lambda x: x.split(".")[0].strip() + x.split(".")[3].strip()
    )
    df["DATE"] = df["Station"].apply(
        lambda x: x.split(".")[1].strip() + "-" + x.split(".")[2].strip()
    )

    df["SIZE"] = df["Station"].apply(
        lambda x: "COARSE" if "10" in x.split(".")[4].strip() else "FINE"
    )
    df["Total"] = df[used_species].sum(axis=1, numeric_only=True)

    df = prep_pah_ad_util(df, "v2")

    return df


def prep_src_prf_pah(roi="delhi"):
    """
    Delhi (2020): Flu, Phe, Ant, Pyr, BaA, Chr, BbF, BkF, BaP, InP, DahA, BghiP
    Seoul (2007): Ind,
    """

    pr_file_path = f"data/source_profile/pah_{roi}.csv"
    u = 0.2 if roi == "delhi" else 0.15

    df = pd.read_csv(pr_file_path)

    tit_cols = ["PNO", "SID", "SIZE"]
    columns = df.columns.tolist()
    used_species = [
        c for c in columns if c not in tit_cols if c not in NOT_USED_SPECIES
    ]
    new_names = {c: f"{c.upper()}C" for c in used_species}

    df["Total"] = df[used_species].sum(axis=1)
    for c in used_species:
        df[c] = df[c] / df["Total"]
        df[f"{c.upper()}U"] = u * df[c]
        df[f"{c.upper()}U"] = df[f"{c.upper()}U"].apply(lambda x: x if x >= 0 else -99)

    df = df.rename(columns=new_names)
    df = df[
        tit_cols + [f"{s.upper()}{n}" for s in used_species for n in ["C", "U"]]
    ].round(4)
    df_fine = df[df["SIZE"] == "FINE"]
    df_coarse = df[df["SIZE"] == "COARSE"]
    df_fine.to_csv(f"data/run/PR_{roi}_fine.csv", index=False)
    df_coarse.to_csv(f"data/run/PR_{roi}_coarse.csv", index=False)
    df.to_csv(f"data/run/PR_{roi}.csv", index=False)
    return df


def prep_src_prof_pah_china():
    pr_file_path = f"data/source_profile/china.csv"

    df = pd.read_csv(pr_file_path)
    columns = df.columns.tolist()
    new_cols = {c: c.upper() for c in columns}
    df = df.rename(columns=new_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols] * 0.01

    df.to_csv("data/run/PR_china_fine.csv", index=False)
    return df


# %%
prep_src_prf_pah()
prep_src_prf_pah("seoul")
prep_src_prof_pah_china()
# %%
