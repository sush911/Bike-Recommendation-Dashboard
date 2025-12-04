import streamlit as st
import pandas as pd

# -------------------- LOAD DATA --------------------
df = pd.read_excel("bike_cleaned.xlsx")
df.columns = df.columns.str.strip()

st.title("Nepal Bike Recommendation System — Full Scoring (BMI + Ergonomics + All Filters)")

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

# -------------------- TERRAIN / PURPOSE MAPPING --------------------
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

# Mandatory: Budget
df_filtered = df_filtered[(df_filtered['Price (Rs)'] >= budget[0]) & 
                          (df_filtered['Price (Rs)'] <= budget[1])]

# OPTIONAL filters become scoring, not elimination! 
# Add feature match flags
for col in ["Terrain Match", "Purpose Match", "User Type Match"]:
    df_filtered[col] = 0

# -------------------- APPLY OPTIONAL MATCH FLAGS --------------------
for i, row in df_filtered.iterrows():
    bike_type = row["Bike Type"]

    if terrain:
        for t in terrain:
            if bike_type in terrain_map.get(t, []):
                df_filtered.at[i, "Terrain Match"] = 1

    if purpose:
        for p in purpose:
            if bike_type in purpose_map.get(p, []):
                df_filtered.at[i, "Purpose Match"] = 1

    if bike_type_input and bike_type in bike_type_input:
        df_filtered.at[i, "User Type Match"] = 1

# -------------------- SCORING FUNCTIONS --------------------
def ergonomic_score(row):
    inseam = rider_height * 0.45
    seat_score = max(0, 100 - abs(row['Seat Height mm'] - inseam) / inseam * 100)
    weight_score = max(0, 100 - abs(row['Kerb Weight mm'] - rider_weight) / rider_weight * 100)
    bmi_score = max(0, 100 - abs(bmi - 22) * 3)
    return 0.4 * seat_score + 0.4 * weight_score + 0.2 * bmi_score

def functional_score(row):
    score = 0
    score += min(row['Engine Displacement']/400 * 20, 20)
    score += min(row['Max power PS']/40 * 20, 20)
    score += min(row['Max Torque By Nm']/35 * 15, 15)
    score += min(row['Ground Clearance mm']/300 * 10, 10)
    score += min(row['Wheel Base mm']/1600 * 10, 10)
    score += min(row['Fuel Tank Litre']/20 * 5, 5)
    tyre_score = ((row['Front Tyres Size width in mm'] * row['Front Tyres Ratio in percentage'] + row['Rear Tyres Size width in mm'] * row['Rear Tyres Ratio in percentage']) / 2) / 200 * 5
    return score + tyre_score

def safety_score(row):
    abs_type = row.get("ABS", "No")
    front_brake = row.get("Front Brake Type", "Drum")
    rear_brake = row.get("Rear Brake Type", "Drum")
    abs_score = 10 if abs_type == "Dual" else 5 if abs_type == "Single" else 0
    brake_score = (5 if front_brake == "Disc" else 0) + (5 if rear_brake == "Disc" else 0)
    return abs_score + brake_score

def preference_score(row):
    return row["Terrain Match"] * 20 + row["Purpose Match"] * 20 + row["User Type Match"] * 20

# -------------------- APPLY SCORING --------------------
if not df_filtered.empty:
    df_filtered["Ergonomic"] = df_filtered.apply(ergonomic_score, axis=1)
    df_filtered["Functional"] = df_filtered.apply(functional_score, axis=1)
    df_filtered["Safety"] = df_filtered.apply(safety_score, axis=1)
    df_filtered["Preference"] = df_filtered.apply(preference_score, axis=1)

    df_filtered["Total Score"] = (
        df_filtered["Ergonomic"] * 0.30 +
        df_filtered["Functional"] * 0.30 +
        df_filtered["Safety"] * 0.20 +
        df_filtered["Preference"] * 0.20
    )

# -------------------- OUTPUT --------------------
st.subheader("Top Recommended Bikes — Full Recommendation Engine")

if df_filtered.empty:
    st.error("No bikes match your budget.")
else:
    top_bikes = df_filtered.sort_values("Total Score", ascending=False).head(15)

    st.dataframe(top_bikes[[
        "Bike Names","Brand","Price (Rs)","Bike Type",
        "Engine Displacement","Max power PS","Max Torque By Nm",
        "Seat Height mm","Kerb Weight mm","Ground Clearance mm",
        "Fuel Tank Litre","Wheel Base mm",
        "ABS","Front Brake Type","Rear Brake Type",
        "Ergonomic","Functional","Safety","Preference","Total Score"
    ]])
