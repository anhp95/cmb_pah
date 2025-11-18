# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

from const import *


import re


def subscript(s):

    replacements = {
        r"\bPM\s?1\.0\b": "PM₁.₀",
        r"\bPM\s?2\.5\b": "PM₂.₅",
        r"\bPM\s?10\b": "PM₁₀",
        r"\bm3\b": "m³",
        r"\bNm3\b": "Nm³",
        r"\bm2\b": "m²",
        r"\bNOx\b": "NOₓ",
        r"\bNO2\b": "NO₂",
        r"\bSO2\b": "SO₂",
        r"\bO3\b": "O₃",
    }

    # Apply each replacement
    for pattern, replacement in replacements.items():
        s = re.sub(pattern, replacement, s)

    return s


def m3_2_Nm3(data, var_col):
    ug_m3 = data[var_col]
    temperature_C = data["Temperature - C"]
    pressure_hPa = data["Pressure - hPa"]
    # Standard conditions
    T0 = 273.15  # K (0°C)
    P0 = 101325  # Pa (1 atm)

    T = temperature_C + 273.15
    pressure_Pa = pressure_hPa * 100

    ug_Nm3 = ug_m3 * (P0 / pressure_Pa) * (T / T0)
    return ug_Nm3


def o2_norm(data, var_col, station_type):

    o2_col = "O2 - %"
    o2_ref = 13 if "ck" in station_type else 6 if "coal" in station_type else None

    return data[var_col] * (20.9 - o2_ref) / (20.9 - data[o2_col])


def prep_col_data(station_type, col_name, row_data):

    # amb
    #     convert to µg/Nm3 done
    #     same standard
    #      CO + O3/ maximum of 8hour (only if 75% data is available)
    # industrial
    #     ck + coal
    #         convert o2 done
    #         unit mg/Nm3 done
    #         STANDARD
    #             ck
    #             coal300
    #             coalgt300
    #     tsp
    #         unit mg/Nm3
    #         standard
    if "amb" in station_type:
        if "µg/m3" in col_name:
            return m3_2_Nm3(row_data, col_name)
        elif "mg/Nm3" in col_name:
            return row_data[col_name] * 1e3
        elif "µg/Nm3" in col_name:
            return row_data[col_name]
    else:
        if "mg/m3" in col_name:
            return m3_2_Nm3(row_data, col_name)
        elif "µg/Nm3" in col_name:
            return row_data[col_name] / 1e3
        elif "mg/Nm3" in col_name:
            return row_data[col_name]

        if "tsp" not in station_type:
            return o2_norm(row_data, col_name, station_type)


def filter_iqr_outliers(df):
    df_filtered = df.copy()

    for col in df.select_dtypes(include="number").columns:
        df_filtered[col] = df_filtered[col].replace(0, np.nan)
        Q1 = df_filtered[col].quantile(0.15)
        Q3 = df_filtered[col].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        lower_bound = lower_bound if lower_bound > 0 else 1e-5

        upper_bound = Q3 + 1.5 * IQR

        mask = (df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)
        df_filtered.loc[mask, col] = np.nan

    return df_filtered


def site_prep(file_path):

    datetime_col = "Datetime"

    df = pd.read_excel(file_path)
    df = df.rename(columns=COLUMN_DICT)
    df = df.drop(columns=["Index"])

    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format="%d/%m/%Y %H:%M")
    except:
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col], format="%H:%M %d/%m/%Y")
        except:
            pass

    columns = df.columns.to_list()

    for c in columns:
        if c != datetime_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = filter_iqr_outliers(df)
    station_type = get_station_type(file_path)
    unit = UNITS[station_type]

    for c in columns:
        for var in POLUTANTS:
            if var in c:
                df[f"{var} - ({unit})"] = df.apply(
                    lambda row: prep_col_data(station_type, c, row), axis=1
                )
                df = df.drop(columns=[c])
    # sort cols
    df = df[sort_col(df.columns)]
    df = df.set_index(datetime_col).sort_index()

    return df


def filter_pollutant_cols(cols):
    return [c for c in cols if any(v in c for v in POLUTANTS)]


def sort_col(cols):
    sorted_cols = ["Datetime"]
    for p in POLUTANTS:
        for c in cols:
            if p in c:
                sorted_cols.append(c)
    for m in METEOS:
        for c in cols:
            if m in c:
                sorted_cols.append(c)

    return sorted_cols


def col2var(col):
    return col.split("-")[0].strip()


def get_station_type(file_path):
    return file_path.split("_")[0].split("\\")[-1]


def resample_with_threshold(df, station_type):
    columns = df.columns
    results = {}

    results["1h"] = df
    # 1H to 8H (≥ 50% of 8 hours = ≥ 4 data points)
    df_1h = df.resample("1h").mean()
    df_8h_mean = df_1h.resample("8h").mean()
    df_8h_count = df_1h.resample("8h").count()
    df_8h = df_8h_mean.where(df_8h_count >= 3)  # 50%

    results["8h"] = df_8h

    # 8H to 24H (≥ 2/3 = 6 of 3*8H = 6)
    df_24h_mean = df_8h.resample("24h").mean()
    df_24h_max = df_8h.resample("24h").max()

    df_24h_count = df_8h.resample("24h").count()
    df_24h = df_24h_mean.where(df_24h_count >= 2)

    # for CO and O3
    if station_type == "amb":
        co_o3_cols = [c for c in columns if "O3" in c or "CO" in c]
        other_cols = [c for c in columns if "O3" not in c and "CO" not in c]

        if len(co_o3_cols) > 0:
            df_co_o3 = df_24h_max.where(df_24h_count >= 2)[co_o3_cols]

            results["24h"] = df_24h[other_cols].merge(
                df_co_o3, left_index=True, right_index=True
            )
    else:
        results["24h"] = df_24h

    # 24H to 1M (≥ 50% of days in month)
    df_1m_mean = df_24h.resample("1m").mean()
    df_1m_count = df_24h.resample("1m").count()
    days_in_month = df_24h.index.to_series().resample("1m").count()  # expected days
    min_valid_days = (days_in_month * 0.5).round()
    mask_month = df_1m_count.ge(min_valid_days.values[:, None])  # align shape
    df_1m = df_1m_mean.where(mask_month)

    results["1m"] = df_1m

    # 1M to 1Y (≥ 6 valid months)
    df_1y_mean = df_1m.resample("1y").mean()
    df_1y_count = df_1m.resample("1y").count()
    mask_year = df_1y_count >= 5  # ≥ 6 months
    df_1y = df_1y_mean.where(mask_year)

    results["1y"] = df_1y

    return results


def get_stats(df, col, threshold, interval, text):
    if interval != "1y":
        above_thresh = df[df[col] > threshold][[col]]
        count_above = above_thresh.shape[0]
        print("-----------------------------------------------------------")
        print(
            f"Total records of {col} for {interval} above {text} {threshold}: {count_above}"
        )
        if count_above > 0:
            above_thresh["month"] = above_thresh.index.month
            above_thresh["day"] = above_thresh.index.date
            above_thresh["hour"] = above_thresh.index.hour

            month_counts = above_thresh["month"].value_counts().sort_index()
            print("Monthly exceedance count:\n", month_counts)

            daily_counts = above_thresh["day"].value_counts().sort_index()
            hourly_counts = above_thresh["hour"].value_counts().sort_index()

            peak_month = month_counts.idxmax()
            peak_date = daily_counts.idxmax()
            peak_hour = hourly_counts.idxmax()

            print(f"Month with most exceedances: {peak_month}")
            print(f"Date with most exceedances: {peak_date}")
            print(f"Hour of day with most exceedances: {peak_hour}")

            fig, axes = plt.subplots(1, 1, figsize=(8, 4))
            month_counts.plot(
                kind="bar",
                title=f"Exceedances per Month of {subscript(col)} - {interval} above {text} standard",
                ax=axes,
            )
            axes.set_xlabel("Month")
            axes.set_ylabel("Count")
            month_names = [calendar.month_abbr[int(m)] for m in month_counts.index]
            axes.set_xticklabels(month_names)

            plt.tight_layout()

    else:
        above_thresh = df[df[col] > threshold][[col]]
        if len(above_thresh) > 0:
            print("-----------------------------------------------------------")
            print(
                f"Total records of {subscript(col)} - {interval} above {text} {threshold}: ",
                len(above_thresh),
            )
            print("Year exceedance count:\n", above_thresh)


def get_raw_stats_1h(list_file):
    agg_funcs = ["min", "mean", "max", "std"]
    dfs = []
    for f in list_file:
        station = f.split("\\")[-1].split(".xlsx")[0]
        station_type = get_station_type(f)

        df = site_prep(f)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

        columns = filter_pollutant_cols(df.columns)
        df = df[columns]

        for interval in ["1h", "8h", "24h"]:

            resampled = df.resample(interval).mean().agg(agg_funcs).T
            resampled = resampled.reset_index().rename(columns={"index": "Pollutant"})

            # Add level to distinguish between the two
            resampled["station_type"] = station_type
            resampled["resample"] = interval
            resampled["station"] = station

            dfs.append(resampled)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.round(2)
    final_df.to_excel("./output/stats.xlsx")

    return final_df


# %%
