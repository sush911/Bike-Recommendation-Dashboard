import streamlit as st
import pandas as pd

# -------------------- LOAD DATA --------------------

df = pd.read_excel("bike_cleaned.xlsx")
df.columns = df.columns.str.strip()  # remove trailing spaces

st.title("Nepal Bike Recommendation System")

# -------------------- USER INPUTS --------------------

rider_height = st.number_input("Enter your height (cm)", 140, 210, 170)
rider_weight = st.number_input("Enter your weight (kg)", 40, 120, 60)

budget = st.slider("Select your budget (Rs)", 200000, 5000000, (200000, 5000000), step=10000)

terrain = st.multiselect("Preferred Terrain (Optional)",
                         ["Urban", "Highway", "Mountain", "Mixed"])

purpose = st.multiselect("Purpose (Optional)",
                         ["Commuting", "Weekend rides", "Off-road", "Touring"])

bike_type_input = st.multiselect("Preferred Bike Type (Optional)", df["Bike Type"].unique())

# -------------------- BMI --------------------

bmi = rider_weight / ((rider_height/100)**2)
st.write(f"Your BMI: {bmi:.2f}")

# -------------------- TERRAIN/PURPOSE MAPPING --------------------

terrain_map = {
    "Urban": ["Commuter", "Streetfighter", "Naked Sport"],
    "Highway": ["Sport", "Cruiser", "Sport-touring"],
    "Mountain": ["Adventure", "Off-road", "Dual-sport"],
    "Mixed": ["Dual-sport", "Streetfighter", "Adventure"]
}

purpose_map = {
    "Commuting": ["Commuter", "Naked Sport"],
    "Weekend rides": ["Streetfighter", "Sport", "Roadster"],
    "Off-road": ["Adventure", "Dual-sport", "Scrambler"],
    "Touring": ["Cruiser", "Sport-touring", "Adventure"]
}

# -------------------- FILTERING --------------------

df_filtered = df.copy()

# Mandatory filter
df_filtered = df_filtered[(df_filtered['Price (Rs)'] >= budget[0]) &
                          (df_filtered['Price (Rs)'] <= budget[1])]

# Optional filters
if terrain:
    allowed_types = set()
    for t in terrain:
        allowed_types.update(terrain_map.get(t, []))
    df_filtered = df_filtered[df_filtered["Bike Type"].isin(allowed_types)]

if purpose:
    allowed_types = set()
    for p in purpose:
        allowed_types.update(purpose_map.get(p, []))
    df_filtered = df_filtered[df_filtered["Bike Type"].isin(allowed_types)]

if bike_type_input:
    df_filtered = df_filtered[df_filtered["Bike Type"].isin(bike_type_input)]

# -------------------- SCORING FUNCTIONS --------------------

def ergonomic_score(row):
    inseam = rider_height * 0.45
    seat_score = max(0, 100 - abs(row['Seat Height mm'] - inseam) / inseam * 100)
    weight_score = max(0, 100 - max(0, (row['Kerb Weight mm'] - rider_weight) / rider_weight * 100))
    return 0.5 * seat_score + 0.5 * weight_score

def functional_score(row):
    score = 0
    score += min(row['Engine Displacement']/400 * 20, 20)
    score += min(row['Max power PS']/40 * 20, 20)
    score += min(row['Max Torque By Nm']/35 * 15, 15)
    score += min(row['Ground Clearance mm']/300 * 10, 10)
    score += min(row['Wheel Base mm']/1600 * 10, 10)
    score += min(row['Fuel Tank Litre']/20 * 5, 5)

    tyre_score = (
        (row['Front Tyres Size width in mm'] * row['Front Tyres Ratio in percentage'] +
         row['Rear Tyres Size width in mm'] * row['Rear Tyres Ratio in percentage']) / 2
    ) / 200 * 5

    return score + tyre_score

def safety_score(row):
    abs_type = row.get("ABS", "No")
    front_brake = row.get("Front Brake Type", "Drum")
    rear_brake = row.get("Rear Brake Type", "Drum")

    abs_score = 10 if abs_type == "Dual" else 5 if abs_type == "Single" else 0
    brake_score = (5 if front_brake == "Disc" else 0) + (5 if rear_brake == "Disc" else 0)

    return abs_score + brake_score

def budget_score(row):
    return 10 if budget[0] <= row['Price (Rs)'] <= budget[1] else 0

# -------------------- APPLY SCORING --------------------

if not df_filtered.empty:
    df_filtered["Ergonomic"] = df_filtered.apply(ergonomic_score, axis=1)
    df_filtered["Functional"] = df_filtered.apply(functional_score, axis=1)
    df_filtered["Safety"] = df_filtered.apply(safety_score, axis=1)
    df_filtered["Budget"] = df_filtered.apply(budget_score, axis=1)

    df_filtered["Total Score"] = (
        df_filtered["Ergonomic"] * 0.35 +
        df_filtered["Functional"] * 0.25 +
        df_filtered["Safety"] * 0.15 +
        df_filtered["Budget"] * 0.10
    )

# -------------------- OUTPUT --------------------

st.subheader("Top Recommended Bikes")

if df_filtered.empty:
    st.error("No bikes match your selected filters. Try removing some optional filters.")
else:
    top_bikes = df_filtered.sort_values("Total Score", ascending=False).head(10)

    st.dataframe(top_bikes[[
        "Bike Names","Brand","Price (Rs)","Bike Type",
        "Engine Displacement","Max power PS","Max power RPM",
        "Max Torque By Nm","Max Torque RPM",
        "Seat Height mm","Kerb Weight mm","Ground Clearance mm",
        "Fuel Tank Litre","Wheel Base mm",
        "Front Brake Type","Rear Brake Type","ABS",
        "Front Tyres Size width in mm","Front Tyres Ratio in percentage",
        "Rear Tyres Size width in mm","Rear Tyres Ratio in percentage"
    ]])
