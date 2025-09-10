# -*- coding: utf-8 -*-
"""Supriyo - EnviroScan Pollution Source Identifier"""

import pandas as pd
import requests
import osmnx as ox
import folium
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# -------------------------------
# Data fetching functions
# -------------------------------
def fetch_openaq_data(city, params):
    url = f"https://api.openaq.org/v2/measurements?city={city}"
    response = requests.get(url, params=params)   # fixed typo (response.get → requests.get)
    data = response.json().get('results', [])
    return pd.DataFrame(data)

def fetch_weather_data(lat, lon, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'lat': lat, 'lon': lon, 'appid': api_key}
    response = requests.get(url, params=params)
    return response.json()

def get_location_features(lat, lon, dist=1000):
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
    roads = ox.geometries.geometries_from_point((lat, lon), tags={'highway': True}, dist=dist)
    factories = ox.geometries.geometries_from_point((lat, lon), tags={'landuse': 'industrial'}, dist=dist)
    return {'roads': roads, 'factories': factories}

# -------------------------------
# Data cleaning & feature engineering
# -------------------------------
def clean_pollution_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['value', 'coordinates.latitude', 'coordinates.longitude'])
    df['value'] = pd.to_numeric(df['value'])
    df['timestamp'] = pd.to_datetime(df['date']['utc'])
    df = df.fillna(df.mean(numeric_only=True))  # fill numeric cols
    return df

def features_engineering(df):
    for col in ['value']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    return df

# -------------------------------
# Source labelling
# -------------------------------
def label_sources(df):
    df['source'] = 'Unknown'
    df.loc[(df['near_main_road'] == 1) & (df['NO2'] > 40), 'source'] = 'Vehicular'
    df.loc[(df['near_factory'] == 1) & (df['SO2'] > 20), 'source'] = 'Industrial'
    df.loc[(df['near_farmland'] == 1) & (df['season'] == 'Dry') & (df['PM2.5'] > 70), 'source'] = 'Agricultural'
    return df

# -------------------------------
# ML model
# -------------------------------
def train_predict_model(df):
    features = ['PM2.5','NO2','SO2','CO','roads_proximity','factories_proximity',
                'temperature','humidity','hour','dayofweek','month']
    X = df[features]
    y = df['source']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    clf = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    st.text("Model Performance:")
    st.text(classification_report(y_test, y_pred))
    return grid.best_estimator_

# -------------------------------
# Map visualization
# -------------------------------
def plot_heatmap(df):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for _, row in df.iterrows():
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=50,
            color="red" if row['source'] == "Industrial" else "blue",
            fill=True
        ).add_to(m)
    return m

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="EnviroScan Pollution Source Identifier")
st.title("🌍 AI-Powered Pollution Source Identifier")

city = st.text_input("Enter a city name", placeholder="e.g., Delhi")

if st.button("Analyze"):
    if city.strip() == "":
        st.warning("Please enter a valid city name.")
    else:
        st.success(f"Analyzing pollution sources for: {city}")

        # ⚠️ Demo output (replace with real data pipeline once APIs are stable)
        st.markdown(f"""
        ### 🔎 AI Analysis Results for **{city}** (Simulated)
        - **Main Pollutants:** PM2.5, NOx, SO2
        - **Likely Sources:**
            - 🚗 Vehicle emissions
            - 🏭 Industrial activity
            - 🔥 Biomass/garbage burning
        - **Air Quality Index (AQI):** 185 (Unhealthy)
        - **Recommendation:** Limit outdoor activity. Use masks. Air purifiers recommended indoors.
        """)
