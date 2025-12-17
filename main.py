import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- LOAD DATA -------------------- #
file_path = "bike_cleaned.xlsx"  # Change this if needed
file_ext = os.path.splitext(file_path)[1].lower()

try:
    if file_ext == '.csv':
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        st.error(f"Unsupported file type: {file_ext}")
        st.stop()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# -------------------- CLEAN DATA -------------------- #
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()

numeric_cols = [
    'Price (Rs)','Engine Displacement','Max power PS','Max power RPM',
    'Max Torque By Nm','Max Torque RPM','Kerb Weight mm','Seat Height mm',
    'Ground Clearance mm','Fuel Tank Litre','Wheel Base mm',
    'Front Tyres Size width in mm','Front Tyres Ratio in percentage',
    'Rear Tyres Size width in mm','Rear Tyres Ratio in percentage',
    'Fuel efficiency'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9\.\-]', '', regex=True), errors='coerce').fillna(0)

# -------------------- STREAMLIT PAGE -------------------- #
st.set_page_config(layout='wide')
st.title("Nepal Bike Recommendation System — Full Thesis Dashboard")

# -------------------- SUMMARY -------------------- #
st.markdown("## Overall Bike Summary")
st.markdown(f"- Total Bikes in Dataset: **{len(df)}**")
st.markdown(f"- Price range: Rs {df['Price (Rs)'].min():,.0f} — Rs {df['Price (Rs)'].max():,.0f}")
st.markdown(f"- Bike Types: {', '.join(sorted(df['Bike Type'].unique()))}")
st.markdown("---")

# -------------------- USER INPUTS -------------------- #
left, mid, right = st.columns([1,1,1])
with left:
    rider_height = st.number_input("Enter your height (cm)", 140, 210, 170)
    rider_weight = st.number_input("Enter your weight (kg)", 40, 120, 60)
    bmi = rider_weight / ((rider_height/100)**2)
    st.write(f"Your BMI: **{bmi:.2f}**")
    st.write("_Ergonomic guide: ideal inseam ~45% of height_")

with mid:
    budget = st.slider("Select your budget (Rs)", int(df['Price (Rs)'].min()), int(max(df['Price (Rs)'].max(),5000000)), (200000, 2000000), step=10000)
    terrain = st.multiselect("Preferred Terrain (Optional)", ["Urban","Highway","Mountain","Mixed"])
    purpose = st.multiselect("Purpose (Optional)", ["Commuting","Weekend rides","Off-road","Touring"]) 

with right:
    brand_filter = st.multiselect("Brand (Optional)", sorted(df['Brand'].unique()))
    abs_filter = st.multiselect("ABS (Optional)", sorted(df['ABS'].unique())) if 'ABS' in df.columns else []
    bike_type_input = st.multiselect("Preferred Bike Type (Optional)", sorted(df['Bike Type'].unique()))

# -------------------- EXTRA NUMERIC FILTERS -------------------- #
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    power_min, power_max = st.slider("Power (PS) range", int(df['Max power PS'].min()), int(df['Max power PS'].max()), (int(df['Max power PS'].min()), int(df['Max power PS'].max())))
with col2:
    torque_min, torque_max = st.slider("Torque (Nm) range", int(df['Max Torque By Nm'].min()), int(df['Max Torque By Nm'].max()), (int(df['Max Torque By Nm'].min()), int(df['Max Torque By Nm'].max())))
with col3:
    weight_max = st.slider("Max kerb weight (kg)", int(df['Kerb Weight mm'].min()), int(df['Kerb Weight mm'].max()), int(df['Kerb Weight mm'].max()))

# -------------------- TERRAIN / PURPOSE MAP -------------------- #
terrain_map = {
    "Urban": ["Commuter","Streetfighter","Naked Sport"],
    "Highway": ["Sport","Fully-faired Sport","Sport-touring","Cruiser","Roadster"],
    "Mountain": ["Adventure","Off-road","Dual-sport","Scrambler"],
    "Mixed": ["Streetfighter","Dual-sport","Adventure"]
}

purpose_map = {
    "Commuting":["Commuter","Naked Sport"],
    "Weekend rides":["Streetfighter","Sport","Roadster"],
    "Off-road":["Adventure","Dual-sport","Scrambler"],
    "Touring":["Cruiser","Sport-touring","Adventure"]
}

# -------------------- FILTER TOGGLE -------------------- #
st.markdown("**Optional filters behaviour**")
filter_toggle = st.checkbox("Apply optional selections as hard filters", value=True)

df_work = df.copy()
df_work = df_work[(df_work['Price (Rs)'] >= budget[0]) & (df_work['Price (Rs)'] <= budget[1])]
df_work = df_work[(df_work['Max power PS'] >= power_min) & (df_work['Max power PS'] <= power_max)]
df_work = df_work[(df_work['Max Torque By Nm'] >= torque_min) & (df_work['Max Torque By Nm'] <= torque_max)]
df_work = df_work[df_work['Kerb Weight mm'] <= weight_max]

if filter_toggle:
    if brand_filter:
        df_work = df_work[df_work['Brand'].isin(brand_filter)]
    if abs_filter:
        df_work = df_work[df_work['ABS'].isin(abs_filter)]
    if bike_type_input:
        df_work = df_work[df_work['Bike Type'].isin(bike_type_input)]
    if terrain:
        allowed = set()
        for t in terrain:
            allowed.update(terrain_map.get(t, []))
        df_work = df_work[df_work['Bike Type'].isin(allowed)]
    if purpose:
        allowed = set()
        for p in purpose:
            allowed.update(purpose_map.get(p, []))
        df_work = df_work[df_work['Bike Type'].isin(allowed)]

# -------------------- OPTIONAL METRIC FILTERS -------------------- #
st.markdown('---')
st.subheader("Optional Additional Metrics Filters")
optional_metrics = ['Seat Height mm', 'Kerb Weight mm', 'Ground Clearance mm', 'Wheel Base mm', 'Fuel Tank Litre']
user_metric_filters = {}
for metric in optional_metrics:
    use_metric = st.checkbox(f"Filter by {metric}?", key=metric)
    if use_metric:
        min_val = int(df_work[metric].min())
        max_val = int(df_work[metric].max())
        sel_range = st.slider(f"Select {metric} range", min_val, max_val, (min_val, max_val))
        user_metric_filters[metric] = sel_range
for metric, (min_v, max_v) in user_metric_filters.items():
    df_work = df_work[(df_work[metric] >= min_v) & (df_work[metric] <= max_v)]

# -------------------- MATCH FLAGS -------------------- #
df_work['Terrain Match'] = 0
df_work['Purpose Match'] = 0
df_work['User Type Match'] = 0
for i, row in df_work.iterrows():
    bt = row['Bike Type']
    if terrain and not filter_toggle:
        for t in terrain:
            if bt in terrain_map.get(t, []):
                df_work.at[i, 'Terrain Match'] = 1
    if purpose and not filter_toggle:
        for p in purpose:
            if bt in purpose_map.get(p, []):
                df_work.at[i, 'Purpose Match'] = 1
    if bike_type_input and not filter_toggle:
        if bt in bike_type_input:
            df_work.at[i, 'User Type Match'] = 1

# -------------------- SCORES -------------------- #
def ergonomic_score(row):
    inseam = rider_height * 0.45
    seat_diff = abs(row['Seat Height mm'] - inseam)
    seat_score = max(0, 100 - (seat_diff / inseam * 100))
    weight_ratio = (row['Kerb Weight mm'] / max(1, rider_weight))
    if weight_ratio <= 0.9:
        weight_score = 100
    elif weight_ratio <= 1.4:
        weight_score = 100 - (weight_ratio - 0.9) / 0.5 * 50
    else:
        weight_score = max(0, 50 - (weight_ratio - 1.4) * 50)
    bmi_score = max(0, 100 - abs(bmi - 22) * 3)
    return 0.45 * seat_score + 0.35 * weight_score + 0.2 * bmi_score

def functional_score(row):
    s = 0
    s += min(row['Engine Displacement'] / 600 * 15, 15)
    s += min(row['Max power PS'] / 60 * 15, 15)
    s += min(row['Max Torque By Nm'] / 50 * 12, 12)
    s += min(row['Max power RPM'] / 12000 * 6, 6)
    s += min(row['Max Torque RPM'] / 9000 * 6, 6)
    s += min(row['Ground Clearance mm'] / 300 * 8, 8)
    s += min(row['Wheel Base mm'] / 1600 * 8, 8)
    s += min(row['Fuel Tank Litre'] / 25 * 5, 5)
    tyre_avg = ((row['Front Tyres Size width in mm'] * row['Front Tyres Ratio in percentage']) +
                (row['Rear Tyres Size width in mm'] * row['Rear Tyres Ratio in percentage'])) / 2
    s += min(tyre_avg / 20000 * 10, 10)
    return s

def safety_score(row):
    abs_type = str(row.get('ABS', 'No'))
    a = 10 if abs_type.lower()=='dual' else 5 if abs_type.lower()=='single' else 0
    fb = 5 if str(row.get('Front Brake Type','')).strip().lower()=='disc' else 0
    rb = 5 if str(row.get('Rear Brake Type','')).strip().lower()=='disc' else 0
    return a + fb + rb

def value_score(row):
    p = row['Price (Rs)']
    low, high = budget[0], budget[1]
    if p < low or p > high: return 0
    return (1 - (p - low) / max(1, (high - low))) * 10

def preference_score(row):
    return row['Terrain Match']*10 + row['Purpose Match']*10 + row['User Type Match']*5

if not df_work.empty:
    df_work['Ergonomic'] = df_work.apply(ergonomic_score, axis=1)
    df_work['Functional'] = df_work.apply(functional_score, axis=1)
    df_work['Safety'] = df_work.apply(safety_score, axis=1)
    df_work['Value'] = df_work.apply(value_score, axis=1)
    df_work['Preference'] = df_work.apply(preference_score, axis=1)
    df_work['Total Score'] = (
        df_work['Ergonomic']*0.30 +
        df_work['Functional']*0.30 +
        df_work['Safety']*0.15 +
        df_work['Value']*0.10 +
        df_work['Preference']*0.15
    )

# -------------------- RECOMMENDATIONS -------------------- #
st.markdown('---')
colA, colB = st.columns([2,1])
with colA:
    st.subheader('Top Recommendations')
    if df_work.empty:
        st.error('No bikes after applying filters. Try widening filters.')
    else:
        top = df_work.sort_values('Total Score', ascending=False).head(15)
        st.dataframe(top)

with colB:
    st.subheader('Controls')
    st.write('Filter toggle:', filter_toggle)

# -------------------- VISUALIZATIONS -------------------- #
st.markdown('---')
st.subheader('Visual Comparisons')
if not df_work.empty:
    top10 = df_work.sort_values('Total Score', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x='Total Score', y='Bike Names', data=top10, color="mediumseagreen")
    ax.set_xlabel('Total Score')
    ax.set_ylabel('Bike Names')
    ax.set_title('Top 10 Bikes by Total Score')
    st.pyplot(fig)

# -------------------- OPTIONAL FEATURE: RADAR COMPARISON -------------------- #
st.markdown('---')
st.subheader("Compare Any 2 Bikes (Radar Chart)")

compare_bikes = st.multiselect("Select two bikes for comparison", df_work['Bike Names'].unique(), default=df_work['Bike Names'].head(2).tolist())

if len(compare_bikes) == 2:
    bike1 = df_work[df_work['Bike Names']==compare_bikes[0]].iloc[0]
    bike2 = df_work[df_work['Bike Names']==compare_bikes[1]].iloc[0]

    categories = ['Ergonomic','Functional','Safety','Value','Preference']
    values1 = [bike1[c] for c in categories]
    values2 = [bike2[c] for c in categories]
    values1 += values1[:1]  # close the loop
    values2 += values2[:1]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax_radar.plot(angles, values1, label=bike1['Bike Names'], linewidth=2)
    ax_radar.fill(angles, values1, alpha=0.25)
    ax_radar.plot(angles, values2, label=bike2['Bike Names'], linewidth=2)
    ax_radar.fill(angles, values2, alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_yticklabels([])
    ax_radar.set_title("Bike Comparison Radar Chart")
    ax_radar.legend(loc='upper right')
    st.pyplot(fig_radar)
