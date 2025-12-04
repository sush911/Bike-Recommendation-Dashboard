import streamlit as st
import pandas as pd

# Load cleaned data
df = pd.read_excel("../data/bikes_cleaned.xlsx")

st.title("Nepal Bike Recommendation System")

# Rider Inputs
rider_height = st.number_input("Enter your height (cm)", 140, 210)
rider_weight = st.number_input("Enter your weight (kg)", 40, 120)
budget = st.slider("Budget (Rs)", int(df['Price (Rs)'].min()), int(df['Price (Rs)'].max()), (200000, 1000000))
terrain = st.selectbox("Terrain you ride on", ["Urban", "Highway", "Mountain", "Mixed"])
purpose = st.selectbox("Purpose", ["Commuting", "Weekend rides", "Off-road", "Touring"])
