import pandas as pd

_data_cache = None

def load_scoring_data():
    global _data_cache
    if _data_cache is None:
        _data_cache = pd.read_csv('city_data.csv')
    return _data_cache

def get_grade(city, year, month):
    df = load_scoring_data()
    match = df[(df['City'] == city) & (df['Year'] == int(year)) & (df['Month'] == int(month))]
    if not match.empty:
        return match.iloc[0]['Grade']
    else:
        return "N/A"

def scale_radar_values(values):
    max_vals = [150, 100, 300, 250, 20]  # max value for each metric based on domain knowledge
    scaled_values = [min(v / m, 1.0) for v, m in zip(values, max_vals)]
    return scaled_values


def get_radar_values(city, year, month):
    df = load_scoring_data()
    match = df[(df['City'] == city) & (df['Year'] == int(year)) & (df['Month'] == int(month))]
    if not match.empty:
        raw_values = [
            match.iloc[0]["Avg_PM2.5_ugm3"],
            match.iloc[0]["Avg_NO2_ppb"],
            match.iloc[0]["Avg_AQI"],
            match.iloc[0]["Rainfall_mm"],
            match.iloc[0]["WindSpeed_ms"]
        ]
        return scale_radar_values(raw_values)
    else:
        return [0, 0, 0, 0, 0]


