# %%
import re
import numpy as np
import pandas as pd


def prep_btex_ad_util(df, version):
    used_species = [
        "Tol",
        "Ben",
        "Eth",
        "Xyl",
    ]

    df["TMA"] = df[used_species].sum(axis=1)
    df["Total"] = df[used_species].sum(axis=1)
    used_species = ["TMA"] + used_species

    for s in used_species:
        df[s] = df[s] / df["Total"]

    species_c = {s: f"{s.upper()}C" for s in used_species}
    species_u = {s: f"{s.upper()}U" for s in used_species}

    keys = ["ID", "DATE", "DUR", "STHOUR", "SIZE"]

    mean_df = (
        df.groupby(["ID", "SIZE", "DATE"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns=species_c)
    )

    std_df = (
        df.groupby(["ID", "SIZE", "DATE"], as_index=False)
        .sem(numeric_only=True)
        .rename(columns=species_u)
    )
    result = pd.merge(mean_df, std_df, on=["ID", "SIZE", "DATE"])
    result["DUR"] = [8] * len(result)
    result["STHOUR"] = [0] * len(result)
    result = result[
        keys + [f"{s.upper()}{n}" for s in used_species for n in ["C", "U"]]
    ].round(4)

    result.to_csv(f"data/run/AD_btex_{version}.csv", index=False)

    return result


def prep_btex_ad():
    ad_file = "./data/ambient/BTEX.dot1.csv"
    df = pd.read_csv(ad_file)

    rename_site = {
        "Background": "BG",
        "Traffic": "TR",
        "Residential": "RS",
    }
    df["Sitetype"] = df["Sitetype"].apply(lambda x: rename_site[x])
    df["DATE"] = df["ID"].apply(lambda x: x[2:][:2] + "/" + x[2:][2:] + "/25")
    df["ID"] = df["ID"].apply(lambda x: re.sub(r"\d+", "", x).strip())
    df["ID"] = df["ID"] + df["Sitetype"]
    df["SIZE"] = "VOC"

    df = prep_btex_ad_util(df, version="v3")
    return df


def prep_btex_src_prf():
    used_species = [
        "Tol",
        "Ben",
        "Eth",
        "Xyl",
    ]
    src_file = "./data/source_profile/btex_china.csv"

    df = pd.read_csv(src_file)
    df["SID"] = df["PNO"]
    df["SIZE"] = "VOC"

    keys = ["PNO", "SID", "SIZE"]

    rename_cols = {c: f"{c.upper()}C" for c in used_species}
    df["Total"] = df[used_species].sum(axis=1)
    for c in used_species:
        df[c] = df[c] / df["Total"]
        df[f"{c.upper()}U"] = 0.2 * df[c]

    df = df.rename(columns=rename_cols)

    df = df[keys + [f"{s.upper()}{n}" for s in used_species for n in ["C", "U"]]].round(
        4
    )
    df.to_csv("./data/run/PR_btex_china.csv", index=False)
    return df


# %%
