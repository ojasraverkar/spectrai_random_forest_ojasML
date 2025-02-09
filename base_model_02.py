import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Using directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "all_indian_districts.csv")

import pandas as pd
import numpy as np
import os

# Load district data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "all_indian_districts.csv")
districts_df = pd.read_csv(csv_path)

# Check for missing values and clean the dataset
if districts_df.isnull().any().any():
    print("Dataset contains missing values. Cleaning the data...")
    districts_df = districts_df.dropna()
    print("Missing values removed.")

# Precompute district coordinates in radians for efficiency
district_coords = np.radians(districts_df[['Latitude', 'Longitude']].values)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    R = 6371  # Radius of Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def get_closest_district(lat, lon):
    """
    Find the closest district to a given latitude and longitude.
    """
    # Convert input latitude and longitude to radians
    lat_lon_radians = np.radians([lat, lon])

    # Calculate distances to all districts
    distances = haversine(lat, lon, districts_df['Latitude'], districts_df['Longitude'])

    # Find the index of the smallest distance
    closest_index = np.argmin(distances)

    # Return the closest district
    return districts_df.iloc[closest_index]["District"]

def train_and_predict(df, use_case, new_data=None):
    """
    Train the model and predict based on the given use case.
    """
    # Define features and target
    categorical_features = ['Optimal Band']
    numerical_features = ['Signal Strength (dBm)', 'Signal Level (0-1)', 
                          'Latitude', 'Longitude', 'Bandwidth (MHz)', 'PCI']

    if use_case == "download":
        target = "Download Speed (Mbps)"
    elif use_case == "upload":
        target = "Upload Speed (Mbps)"
    elif use_case == "ping":
        target = "Ping (ms)"
    else:
        raise ValueError("Invalid use_case")

    # Handle new data
    if new_data is not None:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
    ])

    # Training
    X = df[numerical_features + categorical_features]
    y = df[target]
    model.fit(X, y)

    # Prediction
    all_bands = df['Optimal Band'].unique()
    predictions = []
    for band in all_bands:
        test_row = {**new_data, 'Optimal Band': band}
        X_test = pd.DataFrame([test_row])
        pred = model.predict(X_test)[0]
        predictions.append((band, pred))

    if use_case in ["download", "upload"]:
        best_band = max(predictions, key=lambda x: x[1])
    else:
        best_band = min(predictions, key=lambda x: x[1])

    return best_band[0], best_band[1]

# Main script
try:
    # Load network data
    data = pd.read_csv('simulated_network_data.csv')
    data.columns = data.columns.str.strip()

    # Add District column if missing
    if "District" not in data.columns:
        if "Latitude" in data.columns and "Longitude" in data.columns:
            data["District"] = data.apply(
                lambda row: get_closest_district(row["Latitude"], row["Longitude"]), axis=1
            )
        else:
            raise KeyError("The dataset does not contain 'District' or 'Latitude'/'Longitude' columns.")

    print("Data loaded successfully:")
    print(data.head())

    # Use case and user input
    use_case_input = "download"  # Can be "download", "upload", or "ping"
    user_lat = 12.9236  # Replace with actual latitude
    user_lon = 79.1331  # Replace with actual longitude

    # New speed test data
    new_speedtest = {
        "Signal Strength (dBm)": -75,
        "Signal Level (0-1)": 0.9,
        "Optimal Band": "Band 3",
        "Latitude": user_lat,
        "Longitude": user_lon,
        "Bandwidth (MHz)": 20,
        "PCI": 200,
        "Download Speed (Mbps)": 40,
        "Upload Speed (Mbps)": 15,
        "Ping (ms)": 30,
        "District": get_closest_district(user_lat, user_lon)
    }

    # Train and predict
    best_band, predicted_value = train_and_predict(
        df=data,
        use_case=use_case_input,
        new_data=new_speedtest
    )

    print("\n=== Prediction Results ===")
    print(f"District: {new_speedtest['District']}")
    print(f"Best Band for {use_case_input}: {best_band}")
    print(f"Predicted {use_case_input} Value: {predicted_value:.2f}")

except FileNotFoundError:
    print("Error: 'simulated_network_data.csv' file not found. Please ensure it exists.")
except KeyError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
