import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load your datasets once at module import
case_studies = pd.read_csv("case_s.csv")
case_studies.columns = case_studies.columns.str.strip()
weights = {
    'city_profile': 0.3,
    'driver_cause': 0.35,
    'pollutant': 0.2,
    'intervention': 0.1,
    'temporal': 0.05,
}

# Load city pollution data
df_new = pd.read_csv("city_data.csv")
df_new.columns = df_new.columns.str.strip()

# Convert 'DriverCause' into sets
case_studies['DriverCauseSet'] = case_studies['DriverCause'].apply(lambda x: set(str(x).lower().split(' + ')))
if 'DriverCause' in df_new.columns:
    df_new['DriverCauseSet'] = df_new['DriverCause'].apply(lambda x: set(str(x).lower().split(' + ')))
else:
    df_new['DriverCauseSet'] = [set() for _ in range(len(df_new))]

# Define profile columns
city_profile_columns = ['Population', 'IndustrialIndex', 'UrbanDensity']

# Create dummy profiles if missing
def create_dummy_profiles(df):
    for col in city_profile_columns:
        if col not in df.columns:
            if col == 'Population':
                df[col] = np.random.uniform(1e5, 1e7, size=len(df))
            else:
                df[col] = np.random.uniform(1, 10, size=len(df))
    return df

case_studies = create_dummy_profiles(case_studies)
df_new = create_dummy_profiles(df_new)

# Normalize profiles
scaler = MinMaxScaler()
all_profiles = pd.concat([case_studies[city_profile_columns], df_new[city_profile_columns]], ignore_index=True)
scaler.fit(all_profiles)
case_profiles_norm = scaler.transform(case_studies[city_profile_columns])
df_new_profiles_norm = scaler.transform(df_new[city_profile_columns])


def find_city_date_index(df, city, year, month):
    filtered = df.loc[(df['City'].str.lower() == city.lower()) & (df['Year'] == year) & (df['Month'] == month)]
    if filtered.empty:
        raise ValueError(f"No matching record for {city} {year}-{month}")
    return filtered.index[0]

def score_case_against_query(query_idx, weights):
    query_profile = df_new_profiles_norm[query_idx]
    query_drivers = df_new.at[query_idx, 'DriverCauseSet']
    query_pollutant = df_new.at[query_idx, 'Key_Pollutant'] if 'Key_Pollutant' in df_new.columns else None
    query_intervention_type = df_new.at[query_idx, 'Intervention_Type'] if 'Intervention_Type' in df_new.columns else ''

    scores = []
    for i, row in case_studies.iterrows():
        city_sim = cosine_similarity(query_profile.reshape(1, -1), case_profiles_norm[i].reshape(1, -1))[0][0]
        driver_sim = jaccard_similarity(query_drivers, row['DriverCauseSet'])
        pollutant_sim = 1 if (query_pollutant and query_pollutant == row['Key_Pollutant']) else 0
        intervention_sim = 0
        if query_intervention_type and isinstance(row['Intervention_Type'], str):
            intervention_sim = 1 if query_intervention_type.lower() == row['Intervention_Type'].lower() else 0
        temporal_sim = 1  # placeholder

        total_score = (weights['city_profile'] * city_sim +
                       weights['driver_cause'] * driver_sim +
                       weights['pollutant'] * pollutant_sim +
                       weights['intervention'] * intervention_sim +
                       weights['temporal'] * temporal_sim)
        scores.append((total_score, row))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:3]

def score_case_for_city_date(city, year, month, df_new_arg=None, weights_arg=None):
    df_ref = df_new_arg if df_new_arg is not None else df_new
    used_weights = weights_arg if weights_arg else weights
    query_idx = find_city_date_index(df_ref, city, year, month)
    return score_case_against_query(query_idx, used_weights)

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
