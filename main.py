import streamlit as st
import pandas as pd

# -------------------- LOAD DATA --------------------

df = pd.read_excel("bike_cleaned.xlsx")

# Strip whitespace from column names to avoid KeyErrors
df.columns = df.columns.str.strip()

st.title("Nepal Bike Recommendation System")

# -------------------- USER INPUTS --------------------

rider_height = st.number_input("Enter your height (cm)", min_value=140, max_value=210, value=170)
rider_weight = st.number_input("Enter your weight (kg)", min_value=40, max_value=120, value=60)
budget = st.slider("Select your budget (Rs)", 200000, 5000000, (200000, 5000000), step=10000)
terrain = st.multiselect("Preferred Terrain (Optional)", ["Urban", "Highway", "Mountain", "Mixed"])
purpose = st.multiselect("Purpose of Bike (Optional)", ["Commuting", "Weekend rides", "Off-road", "Touring"])
bike_type_input = st.multiselect("Preferred Bike Type (Optional)", df["Bike Type"].unique())

# -------------------- BMI --------------------

rider_bmi = rider_weight / ((rider_height / 100) ** 2)
st.write(f"Your BMI: {rider_bmi:.2f}")

# -------------------- SCORING FUNCTIONS --------------------

def ergonomic_score(row):
    inseam = rider_height * 0.45
    seat_score = max(0, 100 - abs(row['Seat Height mm'] - inseam) / inseam * 100)
    weight_score = max(0, 100 - max(0, (row['Kerb Weight mm'] - rider_weight) / rider_weight * 100))
    return 0.5 * seat_score + 0.5 * weight_score

def functional_score(row):
    score = 0
    score += min(row['Engine Displacement']/400*20, 20)
    score += min(row['Max power PS']/40*20, 20)
    score += min(row['Max Torque By Nm']/35*15, 15)
    score += min(row['Ground Clearance mm']/300*10, 10)
    score += min(row['Wheel Base mm']/1600*10, 10)
    score += min(row['Fuel Tank Litre']/20*5, 5)
    tyre_score = ((row['Front Tyres Size width in mm']*row['Front Tyres Ratio in percentage'] +
                   row['Rear Tyres Size width in mm']*row['Rear Tyres Ratio in percentage'])/2)/200*5
    score += tyre_score
    return score

def safety_score(row):
    abs_type = row.get('ABS', None)
    front_brake = row.get('Front Brake Type', None)
    rear_brake = row.get('Rear Brake Type', None)
    abs_points = 10 if abs_type == "Dual" else 5 if abs_type == "Single" else 0
    brakes_points = 5 if front_brake == "Disc" else 0
    brakes_points += 5 if rear_brake == "Disc" else 0
    return abs_points + brakes_points

terrain_map = {
    "Urban": ["Commuter", "Streetfighter", "Naked Sport"],
    "Highway": ["Sport", "Fully-faired sportbike", "Sport-touring", "Cruiser", "Roadster"],
    "Mountain": ["Adventure", "Off-road", "Dual-sport", "Scrambler"],
    "Mixed": ["Streetfighter", "Dual-sport", "Adventure"]
}

purpose_map = {
    "Commuting": ["Commuter", "Naked Sport", "Streetfighter"],
    "Weekend rides": ["Streetfighter", "Sport", "Fully-faired sportbike"],
    "Off-road": ["Adventure", "Dual-sport", "Scrambler"],
    "Touring": ["Cruiser", "Sport-touring", "Roadster"]
}

def terrain_purpose_score(row):
    score = 0
    if terrain:
        for t in terrain:
            if row['Bike Type'] in terrain_map[t]:
                score += 10 / len(terrain)
    if purpose:
        for p in purpose:
            if row['Bike Type'] in purpose_map[p]:
                score += 10 / len(purpose)
    if bike_type_input:
        if row['Bike Type'] in bike_type_input:
            score += 5
    return score

def budget_score(row):
    return 10 if budget[0] <= row['Price (Rs)'] <= budget[1] else 0

# -------------------- APPLY SCORES --------------------

filtered_df = df[(df['Price (Rs)'] >= budget[0]) & (df['Price (Rs)'] <= budget[1])].copy()

filtered_df['Ergonomic'] = filtered_df.apply(ergonomic_score, axis=1)
filtered_df['Functional'] = filtered_df.apply(functional_score, axis=1)
filtered_df['Safety'] = filtered_df.apply(safety_score, axis=1)
filtered_df['Terrain_Purpose'] = filtered_df.apply(terrain_purpose_score, axis=1)
filtered_df['Budget'] = filtered_df.apply(budget_score, axis=1)

filtered_df['Total Score'] = (
    filtered_df['Ergonomic'] * 0.35 +
    filtered_df['Functional'] * 0.25 +
    filtered_df['Safety'] * 0.15 +
    filtered_df['Terrain_Purpose'] * 0.15 +
    filtered_df['Budget'] * 0.10
)

# -------------------- DISPLAY TOP BIKES --------------------

top_bikes = filtered_df.sort_values(by='Total Score', ascending=False).head(10)

st.subheader("Top Recommended Bikes")
st.dataframe(top_bikes[['Bike Names','Brand','Price (Rs)','Bike Type','Seat Height mm','Kerb Weight mm',
                        'Engine Displacement','Max power PS','Max power RPM','Max Torque By Nm','Max Torque RPM',
                        'Ground Clearance mm','Fuel Tank Litre','Wheel Base mm','Front Brake Type','Rear Brake Type',
                        'ABS','Front Tyres Size width in mm','Front Tyres Ratio in percentage','Rear Tyres Size width in mm',
                        'Rear Tyres Ratio in percentage','Total Score']])

