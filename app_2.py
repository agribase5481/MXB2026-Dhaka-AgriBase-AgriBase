import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify
import pickle

# --- Configuration and Initialization ---
DATABASE_FILE = 'attempt.db'
app = Flask(__name__)
# Global variables to hold models and encoder
area_model = None
yield_model = None
crop_encoder = None
ALL_CROP_NAMES = []


def load_and_preprocess_data():
    """
    Connects to the SQLite DB, loads all yearly tables, merges them,
    and prepares the data for time-series prediction using lagged features.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # 1. Get all table names (assuming each table is a year's data)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    all_data = []

    # 2. Loop through all tables and load data
    for table_name in table_names:
        # Assuming table names are YEAR_CropData (e.g., '2019_CropData')
        # You may need to adjust this SQL query based on your actual table and column names
        query = f"SELECT * FROM {table_name}"
        try:
            df = pd.read_sql_query(query, conn)
            # Standardize column names (adjust these to match your DB schema exactly)
            df = df.rename(columns={
                'Year': 'Year',
                'CropName': 'Crop',
                'Area_Allocated': 'Area',
                'Yield_Per_Acre': 'Yield'
            })
            all_data.append(df)
        except Exception as e:
            print(f"Error loading table {table_name}: {e}")
            continue

    conn.close()

    if not all_data:
        raise Exception("No data loaded from the database.")

    df_combined = pd.concat(all_data, ignore_index=True)

    # --- Data Cleaning and Feature Engineering ---
    global crop_encoder, ALL_CROP_NAMES

    # Fill any missing values in target columns with their mean
    df_combined['Area'].fillna(df_combined['Area'].mean(), inplace=True)
    df_combined['Yield'].fillna(df_combined['Yield'].mean(), inplace=True)

    # Sort data by crop and year for time-series modeling
    df_combined.sort_values(by=['Crop', 'Year'], inplace=True)

    # 3. Create Lagged Features (Previous Year's Area/Yield)
    # This is how we convert time-series prediction into a regression problem:
    # using past values to predict the next value.
    df_combined['Area_Lag1'] = df_combined.groupby('Crop')['Area'].shift(1)
    df_combined['Yield_Lag1'] = df_combined.groupby('Crop')['Yield'].shift(1)

    # Drop the first row for each crop as Lag1 features will be missing
    df_combined.dropna(subset=['Area_Lag1', 'Yield_Lag1'], inplace=True)

    # 4. Encode the Crop name
    crop_encoder = LabelEncoder()
    df_combined['Crop_Encoded'] = crop_encoder.fit_transform(df_combined['Crop'])
    ALL_CROP_NAMES = list(crop_encoder.classes_)

    return df_combined

def train_models(df):
    """Trains the Area and Yield prediction models."""

    # Features: Encoded Crop name, Previous Year's Area, Previous Year's Yield
    features = ['Crop_Encoded', 'Area_Lag1', 'Yield_Lag1']

    X = df[features]

    # Target 1: Current Year's Area
    Y_area = df['Area']

    # Target 2: Current Year's Yield
    Y_yield = df['Yield']

    # Initialize and train the models
    global area_model, yield_model

    # Model 1: Area Prediction
    area_model = RandomForestRegressor(n_estimators=100, random_state=42)
    area_model.fit(X, Y_area)
    print("Area Model Trained.")

    # Model 2: Yield Prediction
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
    yield_model.fit(X, Y_yield)
    print("Yield Model Trained.")

    return area_model, yield_model


@app.before_first_request
def initialize():
    """Load data and train models once when the app starts."""
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()

        # Save a copy of the last year's actual data for prediction feature input
        last_year_data = df.loc[df.groupby('Crop')['Year'].idxmax()]
        last_year_data[['Crop', 'Area', 'Yield']].to_csv('last_year_data.csv', index=False)

        # Train models
        train_models(df)

    except Exception as e:
        print(f"Initialization failed: {e}")
        # In a real app, you would handle this more gracefully
        global ALL_CROP_NAMES
        ALL_CROP_NAMES = ["Error: Could not load data or train models."]


@app.route('/')
def index():
    """Renders the home page with a list of available crops."""
    return render_template('index_2.html', crop_names=ALL_CROP_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if area_model is None or yield_model is None:
        return jsonify({'error': 'Prediction models not initialized. Check server logs.'})

    try:
        # Get user input
        crop_name = request.form['crop_name']

        # 1. Get the last recorded values for the selected crop
        last_data_df = pd.read_csv('last_year_data.csv')

        # Find the last year's actual area and yield for the chosen crop to use as features
        last_crop_data = last_data_df[last_data_df['Crop'] == crop_name]

        if last_crop_data.empty:
            return jsonify({'error': f'No historical data found for crop: {crop_name}'})

        # Features for the prediction (Lagged Features)
        last_area = last_crop_data['Area'].iloc[0]
        last_yield = last_crop_data['Yield'].iloc[0]

        # Encode the crop name for the model input
        crop_encoded = crop_encoder.transform([crop_name])[0]

        # Create the input array for the models
        prediction_input = np.array([[crop_encoded, last_area, last_yield]])

        # 2. Make Predictions

        # Prediction 1: Area Allocation
        predicted_area = area_model.predict(prediction_input)[0]

        # Prediction 2: Yield Per Acre
        predicted_yield = yield_model.predict(prediction_input)[0]

        # 3. Format and return results
        results = {
            'crop_name': crop_name,
            'predicted_area': f'{predicted_area:,.2f}',
            'predicted_yield': f'{predicted_yield:,.2f}',
            'last_area_input': f'{last_area:,.2f}',
            'last_yield_input': f'{last_yield:,.2f}'
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # The models will be trained on the first request via @app.before_first_request
    # To test locally, ensure your database is in the same directory.
    app.run(debug=True)
