# %%
import pandas as pd
from glob import glob


LIST_XLSX = glob("./data/*.xlsx")

VN_AMB_STD = {
    "SO2": {"1h": 350, "8h": None, "24h": 125, "1y": 50},
    "CO": {"1h": 30000, "8h": None, "24h": 10000, "1y": None},
    "NO2": {"1h": 200, "8h": None, "24h": 100, "1y": 40},
    "O3": {"1h": 200, "8h": None, "24h": 120, "1y": None},
    "TSP": {"1h": 300, "8h": None, "24h": 200, "1y": 100},
    "PM10": {"1h": None, "8h": None, "24h": 100, "1y": 50},
    "PM2.5": {"1h": None, "8h": None, "24h": 50, "1y": 25},
    "PM1.0": {"1h": None, "8h": None, "24h": None, "1y": None},
}

WHO_AMB_STD = {
    "SO2": {"1h": None, "8h": None, "24h": 40, "1y": None},
    "CO": {"1h": None, "8h": None, "24h": 4000, "1y": None},
    "NO2": {"1h": None, "8h": None, "24h": 25, "1y": 10},
    "O3": {"1h": None, "8h": None, "24h": 100, "1y": None},
    "TSP": {"1h": None, "8h": None, "24h": None, "1y": None},
    "PM10": {"1h": None, "8h": None, "24h": 45, "1y": 15},
    "PM2.5": {"1h": None, "8h": None, "24h": 15, "1y": 5},
    "PM1.0": {"1h": None, "8h": None, "24h": None, "1y": None},
}

NON_AMB_STD = {
    "tsp": {"TSP": {"24h": 35}},
    "ck": {
        "TSP": {"24h": 25},
        "SO2": {"24h": 100},
        "NOx": {"24h": 350},
        "CO": {"24h": 250},
    },
    "coal300": {
        "TSP": {"24h": 35},
        "SO2": {"24h": 250},
        "NOx": {"24h": 250},
        "CO": {"24h": 400},
    },
    "coalgt300": {
        "TSP": {"24h": 30},
        "SO2": {"24h": 220},
        "NOx": {"24h": 220},
        "CO": {"24h": 400},
    },
}

UNITS = {
    "amb": "µg/Nm3",
    "tsp": "mg/Nm3",
    "ck": "mg/Nm3",
    "coal300": "mg/Nm3",
    "coalgt300": "mg/Nm3",
}

POLUTANTS = ["SO2", "CO", "NO2", "NOx", "O3", "TSP", "PM10", "PM2.5", "PM1.0"]
METEOS = [
    "Temperature",
    "Radiation",
    "Rainfall",
    "Relative Humidity",
    "Wind Speed",
    "Wind Direction",
    "Pressure",
]

COLUMN_DICT = {
    "STT": "Index",
    "Thời gian": "Datetime",
    "CO (mg/Nm3)": "CO - mg/Nm3",
    "Lưu luợng (m3)": "Flow rate - m3",
    "NOx (mg/Nm3)": "NOx - mg/Nm3",
    "O2 (%)": "O2 - %",
    "Press (hPa)": "Pressure - hPa",
    "Bụi (mg/Nm3)": "TSP - mg/Nm3",
    "SO2 (mg/Nm3)": "SO2 - mg/Nm3",
    "Temp (oC)": "Temperature - C",
    "ICO (ug/m3)": "CO - µg/m3",
    "INO (ug/m3)": "NO - µg/m3",
    "INO2 (ug/m3)": "NO2 - µg/m3",
    "INOx (ppb)": "NOIx - ppb",
    "IO3 (ug/m3)": "O3 - µg/m3",
    "IPM1 (µg/m3)": "PM1.0 - µg/m3",
    "IPM10 (µg/m3)": "PM10 - µg/m3",
    "IPM2 (µg/m3)": "PM2.5 - µg/m3",
    "ISO2 (ug/m3)": "SO2 - µg/m3",
    "ITSP (ug/m3)": "TSP - µg/m3",
    "MCompass (Degree)": "Compass Direction - Degree",
    "MGlob (W/m2)": "Radiation - W/m2",
    "MRain (mm/h)": "Rainfall - mm/h",
    "MTemp (oC)": "Temperature - C",
    "MWindDir (Degree)": "Wind Direction - Degree",
    "MWindSpeed (m/s)": "Wind Speed - m/s",
    "Mamb (hPa)": "Pressure - hPa",
    "Mrel (%)": "Relative Humidity - %",
    "Datetime": "Datetime",
    "Dust(mg/Nm3}": "TSP - mg/Nm3",
    "NOx(µg/Nm3}": "NOx - µg/Nm3",
    "O2(%}": "O2 - %",
    "SO2(µg/Nm3}": "SO2 - µg/Nm3",
    "Temp(oC}": "Temperature - C",
    "Flow(m3/h}": "Flow rate - m3/h",
    "Nhiệt độ(oC}": "Temperature - C",
    "PM-10(µg/Nm3}": "PM10 - µg/Nm3",
    "PM-2-5(µg/Nm3}": "PM2.5 - µg/Nm3",
    "NO(µg/Nm3}": "NO - µg/Nm3",
    "NO2(µg/Nm3}": "NO2 - µg/Nm3",
    "InnerTemp(oC}": "IT - C",
    "Radiation(W/m2}": "Radiation - W/m2",
    "Hướng gió(Degree}": "Wind Direction - Degree",
    "Tốc độ gió(m/s}": "Wind Speed - m/s",
    "Độ ẩm(%}": "Relative Humidity - %",
    "Áp suất khí quyển(hPa}": "Pressure - hPa",
    "PM-1(µg/m3}": "PM1.0 - µg/m3",
    "RH(%}": "Relative Humidity - %",
}


def flatten_unique(lst):
    seen = set()
    result = []

    def helper(sublist):
        for item in sublist:
            if isinstance(item, list):
                helper(item)
            elif item not in seen:
                seen.add(item)
                result.append(item)

    helper(lst)
    return result


def read_xls(list_files):
    unique_cols = [pd.read_excel(f).columns.to_list() for f in list_files]
    unique_cols = flatten_unique(unique_cols)
    return unique_cols


# %%
