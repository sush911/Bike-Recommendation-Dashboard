import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- LOAD & CLEAN DATA -------------------- #
df = pd.read_excel("bike_cleaned.xlsx")

# Clean column names and string values
df.columns = df.columns.str.strip()
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()

# Numeric columns
num_cols = ['Price (Rs)','Engine Displacement','Max power PS','Max power RPM','Max Torque By Nm','Max Torque RPM',
            'Kerb Weight mm','Seat Height mm','Ground Clearance mm','Fuel Tank Litre','Wheel Base mm',
            'Front Tyres Size width in mm','Front Tyres Ratio in percentage','Rear Tyres Size width in mm','Rear Tyres Ratio in percentage']

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9\.\-]', '', regex=True), errors='coerce').fillna(0)

st.set_page_config(layout='wide')
st.title("Nepal Bike Recommendation System — Full Feature Dashboard")

# -------------------- USER INPUTS / FILTERS -------------------- #
left, mid, right = st.columns([1,1,1])
with left:
    rider_height = st.number_input("Enter your height (cm)", 140, 210, 170)
    rider_weight = st.number_input("Enter your weight (kg)", 40, 120, 60)
    bmi = rider_weight / ((rider_height/100)**2)
    st.write(f"Your BMI: **{bmi:.2f}**")
    st.write("_Ergonomic quick guide: ideal inseam ~45% of height_")

with mid:
    budget = st.slider("Select your budget (Rs)", int(df['Price (Rs)'].min()), int(max(df['Price (Rs)'].max(),5000000)),
                       (200000, 2000000), step=10000)
    terrain = st.multiselect("Preferred Terrain (Optional)", ["Urban","Highway","Mountain","Mixed"])
    purpose = st.multiselect("Purpose (Optional)", ["Commuting","Weekend rides","Off-road","Touring"]) 

with right:
    brand_filter = st.multiselect("Brand (Optional)", sorted(df['Brand'].unique()))
    abs_filter = st.multiselect("ABS (Optional)", sorted(df['ABS'].unique())) if 'ABS' in df.columns else []
    bike_type_input = st.multiselect("Preferred Bike Type (Optional)", sorted(df['Bike Type'].unique()))

# Extra numeric filters
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    power_min, power_max = st.slider("Power (PS) range", int(df['Max power PS'].min()), int(df['Max power PS'].max()),
                                    (int(df['Max power PS'].min()), int(df['Max power PS'].max())))
with col2:
    torque_min, torque_max = st.slider("Torque (Nm) range", int(df['Max Torque By Nm'].min()), int(df['Max Torque By Nm'].max()),
                                       (int(df['Max Torque By Nm'].min()), int(df['Max Torque By Nm'].max())))
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
st.markdown("**Optional filters behaviour**: choose whether optional selections should filter or just influence score")
filter_toggle = st.checkbox("Apply optional selections as hard filters (terrain/purpose/bike type/brand/ABS)", value=True)

df_work = df.copy()
# Mandatory filters
df_work = df_work[(df_work['Price (Rs)'] >= budget[0]) & (df_work['Price (Rs)'] <= budget[1])]
df_work = df_work[(df_work['Max power PS'] >= power_min) & (df_work['Max power PS'] <= power_max)]
df_work = df_work[(df_work['Max Torque By Nm'] >= torque_min) & (df_work['Max Torque By Nm'] <= torque_max)]
df_work = df_work[df_work['Kerb Weight mm'] <= weight_max]

# Optional hard filters
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

# Apply optional metric filters
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
    return 0.45 * seat_score + 0.35 * weight_score + 0.20 * bmi_score

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
    st.write('Top selected:', 10)
    st.write('Filter toggle:', filter_toggle)

# -------------------- VISUALIZATIONS -------------------- #
st.markdown('---')
st.subheader('Visual Comparisons')
if not df_work.empty:
    top10 = df_work.sort_values('Total Score', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(top10['Bike Names'][::-1], top10['Total Score'][::-1])
    ax.set_xlabel('Total Score')
    ax.set_title('Top 10 Bikes by Total Score')
    st.pyplot(fig)

st.markdown('---')
st.subheader('Bike Detail - Radar (Ergonomic / Functional / Safety / Value / Preference)')
bike_options = list(df_work['Bike Names'].unique())
if bike_options:
    selected = st.selectbox('Select a bike for detailed view', bike_options)
    b = df_work[df_work['Bike Names']==selected].iloc[0]
    labels = ['Ergonomic','Functional','Safety','Value','Preference']
    values = [b['Ergonomic'], b['Functional'], b['Safety'], b['Value'], b['Preference']]
    max_vals = [100,100,20,10,20]
    vals_norm = [v/m if m>0 else 0 for v,m in zip(values,max_vals)]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals_wrap = vals_norm + vals_norm[:1]
    angles = angles + angles[:1]
    fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax2.plot(angles, vals_wrap, 'o-', linewidth=2)
    ax2.fill(angles, vals_wrap, alpha=0.25)
    ax2.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax2.set_title(selected)
    st.pyplot(fig2)

# -------------------- COMPARE TWO BIKES -------------------- #
st.markdown('---')
st.subheader('Compare Two Bikes')
compare1 = st.selectbox('Bike A', bike_options, index=0 if len(bike_options)>0 else None)
compare2 = st.selectbox('Bike B', bike_options, index=1 if len(bike_options)>1 else None)
if compare1 and compare2 and compare1 != compare2:
    A = df_work[df_work['Bike Names']==compare1].iloc[0]
    B = df_work[df_work['Bike Names']==compare2].iloc[0]
    compare_df = pd.DataFrame({
        'Metric': ['Price (Rs)','Engine Displacement','Max power PS','Max power RPM','Max Torque By Nm','Max Torque RPM',
                   'Kerb Weight mm','Seat Height mm','Ground Clearance mm','Fuel Tank Litre','Wheel Base mm','ABS','Front Brake Type','Rear Brake Type','Total Score'],
        'Bike A': [A['Price (Rs)'], A['Engine Displacement'], A['Max power PS'], A['Max power RPM'], A['Max Torque By Nm'], A['Max Torque RPM'],
                   A['Kerb Weight mm'], A['Seat Height mm'], A['Ground Clearance mm'], A['Fuel Tank Litre'], A['Wheel Base mm'], A['ABS'], A['Front Brake Type'], A['Rear Brake Type'], A['Total Score']],
        'Bike B': [B['Price (Rs)'], B['Engine Displacement'], B['Max power PS'], B['Max power RPM'], B['Max Torque By Nm'], B['Max Torque RPM'],
                   B['Kerb Weight mm'], B['Seat Height mm'], B['Ground Clearance mm'], B['Fuel Tank Litre'], B['Wheel Base mm'], B['ABS'], B['Front Brake Type'], B['Rear Brake Type'], B['Total Score']]
    })
    st.dataframe(compare_df)

# -------------------- ERGONOMIC FIT INSIGHTS -------------------- #
st.markdown('---')
st.subheader('Ergonomic Fit Insights')
if not df_work.empty:
    df_work['Seat Diff (cm)'] = abs(df_work['Seat Height mm'] - rider_height*0.45)
    df_work['Seat Fit Category'] = pd.cut(df_work['Seat Diff (cm)'], bins=[-1,5,15,1000], labels=['Good Fit','Acceptable','Poor Fit'])
    st.write('Distribution of seat fit categories:')
    st.dataframe(df_work[['Bike Names','Seat Height mm','Seat Diff (cm)','Seat Fit Category']].sort_values('Seat Diff (cm)').head(20))

# -------------------- WEIGHT / HANDLING CATEGORY -------------------- #
st.markdown('---')
st.subheader('Weight / Handling Category')
if not df_work.empty:
    def weight_category(row):
        ratio = row['Kerb Weight mm'] / max(1, rider_weight)
        if ratio <= 1.05:
            return 'Very Easy (Beginner)'
        elif ratio <= 1.2:
            return 'Easy (Friendly)'
        elif ratio <= 1.4:
            return 'Moderate'
        else:
            return 'Heavy (Experienced)'
    df_work['Weight Category'] = df_work.apply(weight_category, axis=1)
    st.dataframe(df_work[['Bike Names','Kerb Weight mm','Weight Category']].sort_values('Kerb Weight mm').head(30))

# -------------------- EXPORT / REPORT -------------------- #
st.markdown('---')
st.subheader('Export')
if not df_work.empty:
    if st.button('Export top 10 to CSV'):
        out = df_work.sort_values('Total Score', ascending=False).head(10)
        out.to_csv('top_bikes.csv', index=False)
        st.success('Exported top_bikes.csv in current working directory')
