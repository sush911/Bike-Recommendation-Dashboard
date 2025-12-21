import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== CONFIGURATION ==================== #
st.set_page_config(layout='wide', page_title="Nepal Bike Recommendation System")

# ==================== LOAD DATA ==================== #
@st.cache_data
def load_data(file_path):
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
    
    # Clean column names and string data
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().str.replace('"','').str.replace("'",'')
    
    # Handle Kerb Weight column naming
    if 'Kerb Weight mm' in df.columns:
        df.rename(columns={'Kerb Weight mm':'Kerb Weight'}, inplace=True)
    elif 'Kerb Weight kg' in df.columns:
        df.rename(columns={'Kerb Weight kg':'Kerb Weight'}, inplace=True)
    
    # Clean numeric columns
    numeric_cols = [
        'Price (Rs)','Engine Displacement','Max power PS','Max power RPM',
        'Max Torque By Nm','Max Torque RPM','Kerb Weight','Seat Height mm',
        'Ground Clearance mm','Fuel Tank Litre','Wheel Base mm',
        'Front Tyres Size width in mm','Front Tyres Ratio in percentage',
        'Rear Tyres Size width in mm','Rear Tyres Ratio in percentage',
        'Fuel efficiency'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9\.\-]', '', regex=True), errors='coerce').fillna(0)
    
    return df

file_path = "bike_cleaned.xlsx"
df = load_data(file_path)

# ==================== TITLE AND METHODOLOGY ==================== #
st.title("🏍️ Nepal Bike Recommendation System")
st.markdown("### Specification-Based Recommendation for Young Adults (18-35)")

with st.expander("📊 **Research Methodology & System Logic**"):
    st.markdown("""
    **Research Questions:**
    1. How do bike specifications influence functional suitability for young Nepali riders?
    2. How do rider height, weight, and BMI affect ergonomic fit and safety?
    
    **Hypotheses:**
    - H1: Matching bike specs with rider characteristics improves suitability vs brand-based choices
    - H2: Data-driven recommendations reduce social/marketing bias in motorcycle selection
    
    **Scoring System Components:**
    
    **1. Ergonomic Score (30%)** - Rider-bike physical compatibility
    - Seat Height Fit (60%): seat height vs inseam (ideal: ≤90% of inseam)
    - Weight Handling (40%): bike/rider weight ratio for control at stops
    - BMI calculated for reference but not scored (acknowledged limitation)
    
    **2. Functional Score (30%)** - Performance specifications
    - Engine: displacement, power, torque (40%)
    - Fuel Economy (5%): fuel efficiency for cost-effectiveness
    - Chassis: ground clearance, wheelbase, fuel capacity (30%)
    - Tires: width and aspect ratio (15%)
    - RPM characteristics (10%)
    
    **3. Safety Score (20%)** - Objective safety features
    - ABS type: Dual/Single/None (50%)
    - Brake types: Disc/Drum front and rear (50%)
    
    **4. Terrain/Purpose Match Score (10%)** - Use-case suitability
    - Matches bike type to intended terrain and riding purpose
    - Based on bike specifications and categorical data
    
    **5. Value Score (10%)** - Price optimization
    - Rewards bikes in 30-60% budget range (sweet spot)
    
    **Addressing Research Bias:**
    - Brand filter available but flagged as potential marketing bias source
    - Terrain/purpose matching uses bike type data to answer RQ1 about specs
    - System prioritizes measurable specifications over subjective preferences
    - **NEW**: Fuel efficiency integrated into functional scoring for real-world cost considerations
    
    **Acknowledged Limitations:**
    - Weight ratios use simplified thresholds (need empirical validation)
    - Missing data: handlebar reach, knee angles, footpeg position, riding position lean
    - Normalization based on dataset range, not universal engineering standards
    - Terrain mapping based on typical bike type characteristics (needs validation)
    """)

# ==================== DATASET SUMMARY ==================== #
st.markdown("---")
st.markdown("## 📈 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Bikes", len(df))
with col2:
    st.metric("Price Range", f"Rs {df['Price (Rs)'].min():,.0f} - {df['Price (Rs)'].max():,.0f}")
with col3:
    st.metric("Bike Types", len(df['Bike Type'].unique()))
with col4:
    st.metric("Brands", len(df['Brand'].unique()))

# Show available bike types and their counts
with st.expander("🏍️ **Bike Types in Dataset**"):
    bike_type_counts = df['Bike Type'].value_counts()
    st.dataframe(bike_type_counts.reset_index().rename(columns={'index': 'Bike Type', 'Bike Type': 'Count'}))

# ==================== USER INPUTS ==================== #
st.markdown("---")
st.markdown("## 👤 Rider Physical Characteristics")

col_left, col_mid, col_right = st.columns([1,1,1])

with col_left:
    rider_height_cm = st.number_input("Height (cm)", 140, 210, 170, help="Your height in centimeters")
    rider_height_mm = rider_height_cm * 10  # Convert to mm for calculations
    
with col_mid:
    rider_weight = st.number_input("Weight (kg)", 40, 120, 65, help="Your weight in kilograms")
    
with col_right:
    bmi = rider_weight / ((rider_height_cm/100)**2)
    st.metric("Calculated BMI", f"{bmi:.2f}")
    st.caption("BMI shown for reference only")

# Calculate inseam
inseam_mm = rider_height_mm * 0.45
inseam_cm = inseam_mm / 10

st.info(f"📏 **Estimated Inseam: {inseam_cm:.1f} cm ({inseam_mm:.0f} mm)** - Ideal seat height: ≤ {inseam_cm * 0.9:.1f} cm for flat-foot reach")

# ==================== RIDING CONTEXT ==================== #
st.markdown("---")
st.markdown("## 🎯 Riding Purpose & Terrain (Answers RQ1: Spec Influence)")

terrain_purpose_col1, terrain_purpose_col2 = st.columns(2)

# Terrain mapping based on bike specifications
TERRAIN_BIKE_MAP = {
    "Urban/City": {
        "types": ["Commuter", "Scooter", "Naked Sport", "Streetfighter"],
        "specs": "Low seat height, light weight, fuel efficient, easy maneuverability"
    },
    "Highway/Touring": {
        "types": ["Sport", "Fully-faired Sport", "Sport-touring", "Cruiser", "Roadster"],
        "specs": "Higher power, larger fuel tank, comfortable ergonomics, windscreen"
    },
    "Mountain/Off-road": {
        "types": ["Adventure", "Off-road", "Dual-sport", "Scrambler", "Enduro"],
        "specs": "High ground clearance, long suspension travel, knobby tires, light weight"
    },
    "Mixed/Versatile": {
        "types": ["Adventure", "Dual-sport", "Streetfighter", "Naked Sport"],
        "specs": "Balanced specs, moderate seat height, good ground clearance"
    }
}

PURPOSE_BIKE_MAP = {
    "Daily Commuting": {
        "types": ["Commuter", "Scooter", "Naked Sport"],
        "specs": "Fuel efficient, low maintenance, comfortable seat, easy to ride"
    },
    "Weekend/Sport Riding": {
        "types": ["Sport", "Fully-faired Sport", "Streetfighter", "Naked Sport", "Roadster"],
        "specs": "High power, responsive handling, aggressive ergonomics"
    },
    "Long Distance Touring": {
        "types": ["Sport-touring", "Adventure", "Cruiser", "Tourer"],
        "specs": "Large fuel tank, comfortable seat, luggage capacity, windscreen"
    },
    "Off-road/Trail": {
        "types": ["Off-road", "Dual-sport", "Enduro", "Scrambler"],
        "specs": "High ground clearance, knobby tires, light weight, long suspension"
    },
    "Mixed Use": {
        "types": ["Adventure", "Dual-sport", "Streetfighter", "Standard"],
        "specs": "Versatile specs, balanced power/comfort, moderate weight"
    }
}

with terrain_purpose_col1:
    st.markdown("### Preferred Terrain")
    selected_terrains = st.multiselect(
        "Where will you ride most?",
        options=list(TERRAIN_BIKE_MAP.keys()),
        help="Matches bikes based on terrain-appropriate specifications"
    )
    
    if selected_terrains:
        for terrain in selected_terrains:
            with st.expander(f"ℹ️ {terrain}"):
                st.write(f"**Suitable bike types:** {', '.join(TERRAIN_BIKE_MAP[terrain]['types'])}")
                st.write(f"**Key specs:** {TERRAIN_BIKE_MAP[terrain]['specs']}")

with terrain_purpose_col2:
    st.markdown("### Riding Purpose")
    selected_purposes = st.multiselect(
        "What's your primary use?",
        options=list(PURPOSE_BIKE_MAP.keys()),
        help="Matches bikes to your riding style and needs"
    )
    
    if selected_purposes:
        for purpose in selected_purposes:
            with st.expander(f"ℹ️ {purpose}"):
                st.write(f"**Suitable bike types:** {', '.join(PURPOSE_BIKE_MAP[purpose]['types'])}")
                st.write(f"**Key specs:** {PURPOSE_BIKE_MAP[purpose]['specs']}")

# ==================== BUDGET & CORE FILTERS ==================== #
st.markdown("---")
st.markdown("## 💰 Budget & Core Specifications")

col1, col2 = st.columns(2)
with col1:
    budget_min = int(df['Price (Rs)'].min())
    budget_max = int(df['Price (Rs)'].max())
    budget = st.slider("Budget Range (Rs)", budget_min, budget_max, (200000, 2000000), step=50000)

with col2:
    max_weight_limit = st.slider("Maximum Bike Weight (kg)", 
                                  int(df['Kerb Weight'].min()), 
                                  int(df['Kerb Weight'].max()), 
                                  int(df['Kerb Weight'].max()),
                                  help="Heavier bikes require more strength to handle")

st.markdown("### Performance Range Filters")
perf_col1, perf_col2, perf_col3 = st.columns(3)
with perf_col1:
    power_range = st.slider("Power (PS)", 
                            int(df['Max power PS'].min()), 
                            int(df['Max power PS'].max()), 
                            (int(df['Max power PS'].min()), int(df['Max power PS'].max())))
with perf_col2:
    torque_range = st.slider("Torque (Nm)", 
                             int(df['Max Torque By Nm'].min()), 
                             int(df['Max Torque By Nm'].max()), 
                             (int(df['Max Torque By Nm'].min()), int(df['Max Torque By Nm'].max())))
with perf_col3:
    displacement_range = st.slider("Engine Displacement (cc)", 
                                    int(df['Engine Displacement'].min()), 
                                    int(df['Engine Displacement'].max()), 
                                    (int(df['Engine Displacement'].min()), int(df['Engine Displacement'].max())))

# ==================== ADDITIONAL SPEC FILTERS ==================== #
st.markdown("### Additional Specification Filters")
add_spec_col1, add_spec_col2, add_spec_col3 = st.columns(3)

with add_spec_col1:
    if 'Ground Clearance mm' in df.columns:
        use_gc_filter = st.checkbox("Filter Ground Clearance", value=False)
        if use_gc_filter:
            gc_range = st.slider("Ground Clearance (mm)",
                                int(df['Ground Clearance mm'].min()),
                                int(df['Ground Clearance mm'].max()),
                                (int(df['Ground Clearance mm'].min()), int(df['Ground Clearance mm'].max())))
        else:
            gc_range = None
    else:
        gc_range = None

with add_spec_col2:
    if 'Fuel Tank Litre' in df.columns:
        use_fuel_filter = st.checkbox("Filter Fuel Tank", value=False)
        if use_fuel_filter:
            fuel_range = st.slider("Fuel Tank Capacity (L)",
                                  int(df['Fuel Tank Litre'].min()),
                                  int(df['Fuel Tank Litre'].max()),
                                  (int(df['Fuel Tank Litre'].min()), int(df['Fuel Tank Litre'].max())))
        else:
            fuel_range = None
    else:
        fuel_range = None

with add_spec_col3:
    if 'Fuel efficiency' in df.columns:
        use_fe_filter = st.checkbox("Filter Fuel Efficiency", value=False)
        if use_fe_filter:
            fe_range = st.slider("Fuel Efficiency (kmpl)",
                                int(df['Fuel efficiency'].min()),
                                int(df['Fuel efficiency'].max()),
                                (int(df['Fuel efficiency'].min()), int(df['Fuel efficiency'].max())))
        else:
            fe_range = None
    else:
        fe_range = None

# ==================== OPTIONAL FILTERS ==================== #
st.markdown("---")
st.markdown("## 🔧 Optional Filters")
st.warning("⚠️ **Research Note**: Brand filtering may introduce marketing bias (H2 concern). Use cautiously.")

opt_col1, opt_col2, opt_col3 = st.columns(3)

with opt_col1:
    use_brand_filter = st.checkbox("Filter by Brand", value=False)
    brand_filter = []
    if use_brand_filter:
        brand_filter = st.multiselect("Select Brands", sorted(df['Brand'].unique()))

with opt_col2:
    use_abs_filter = st.checkbox("Filter by ABS", value=False)
    abs_filter = []
    if use_abs_filter and 'ABS' in df.columns:
        abs_filter = st.multiselect("ABS Type", sorted(df['ABS'].unique()))

with opt_col3:
    use_type_filter = st.checkbox("Filter by Bike Type", value=False)
    type_filter = []
    if use_type_filter:
        type_filter = st.multiselect("Bike Types", sorted(df['Bike Type'].unique()))

# ==================== APPLY FILTERS ==================== #
df_filtered = df.copy()

# Core filters (always applied)
df_filtered = df_filtered[
    (df_filtered['Price (Rs)'] >= budget[0]) & 
    (df_filtered['Price (Rs)'] <= budget[1]) &
    (df_filtered['Max power PS'] >= power_range[0]) & 
    (df_filtered['Max power PS'] <= power_range[1]) &
    (df_filtered['Max Torque By Nm'] >= torque_range[0]) & 
    (df_filtered['Max Torque By Nm'] <= torque_range[1]) &
    (df_filtered['Engine Displacement'] >= displacement_range[0]) & 
    (df_filtered['Engine Displacement'] <= displacement_range[1]) &
    (df_filtered['Kerb Weight'] <= max_weight_limit)
]

# Additional spec filters
if gc_range:
    df_filtered = df_filtered[(df_filtered['Ground Clearance mm'] >= gc_range[0]) & 
                              (df_filtered['Ground Clearance mm'] <= gc_range[1])]
if fuel_range:
    df_filtered = df_filtered[(df_filtered['Fuel Tank Litre'] >= fuel_range[0]) & 
                              (df_filtered['Fuel Tank Litre'] <= fuel_range[1])]
if fe_range:
    df_filtered = df_filtered[(df_filtered['Fuel efficiency'] >= fe_range[0]) & 
                              (df_filtered['Fuel efficiency'] <= fe_range[1])]

# Optional filters
if use_brand_filter and brand_filter:
    df_filtered = df_filtered[df_filtered['Brand'].isin(brand_filter)]
if use_abs_filter and abs_filter:
    df_filtered = df_filtered[df_filtered['ABS'].isin(abs_filter)]
if use_type_filter and type_filter:
    df_filtered = df_filtered[df_filtered['Bike Type'].isin(type_filter)]

st.info(f"🔍 **Bikes after filtering: {len(df_filtered)}** (from {len(df)} total)")

# ==================== NORMALIZATION ==================== #
def normalize_column(df, col, max_val=None):
    """Normalize column to 0-100 scale"""
    if col not in df.columns:
        return df
    if max_val is None:
        max_val = df[col].max()
    if max_val == 0:
        df[col + '_Norm'] = 0
    else:
        df[col + '_Norm'] = (df[col] / max_val * 100).clip(0, 100)
    return df

if not df_filtered.empty:
    metrics_to_normalize = [
        'Engine Displacement', 'Max power PS', 'Max Torque By Nm', 
        'Max power RPM', 'Max Torque RPM', 'Ground Clearance mm',
        'Wheel Base mm', 'Fuel Tank Litre', 'Kerb Weight', 'Fuel efficiency',
        'Front Tyres Size width in mm', 'Rear Tyres Size width in mm'
    ]
    
    for metric in metrics_to_normalize:
        if metric in df_filtered.columns:
            df_filtered = normalize_column(df_filtered, metric)

# ==================== SCORING FUNCTIONS ==================== #

def calculate_ergonomic_score(row, rider_height_mm, rider_weight, inseam_mm):
    """Ergonomic scoring based on rider-bike physical compatibility"""
    score = 0
    
    # 1. SEAT HEIGHT FIT (60 points)
    seat_height_mm = row.get('Seat Height mm', 0)
    if seat_height_mm > 0 and inseam_mm > 0:
        ideal_max_seat = inseam_mm * 0.90
        
        if seat_height_mm <= ideal_max_seat:
            seat_score = 60
        elif seat_height_mm <= inseam_mm:
            excess_ratio = (seat_height_mm - ideal_max_seat) / (inseam_mm * 0.10)
            seat_score = 60 - (excess_ratio * 30)
        else:
            seat_score = max(0, 30 - ((seat_height_mm - inseam_mm) / inseam_mm * 100))
    else:
        seat_score = 30
    
    # 2. WEIGHT HANDLING (40 points)
    bike_weight = row.get('Kerb Weight', 0)
    if bike_weight > 0 and rider_weight > 0:
        weight_ratio = bike_weight / rider_weight
        
        if weight_ratio <= 2.0:
            weight_score = 40
        elif weight_ratio <= 2.5:
            weight_score = 40 - ((weight_ratio - 2.0) / 0.5 * 15)
        elif weight_ratio <= 3.0:
            weight_score = 25 - ((weight_ratio - 2.5) / 0.5 * 15)
        else:
            weight_score = max(0, 10 - ((weight_ratio - 3.0) * 5))
    else:
        weight_score = 20
    
    score = seat_score + weight_score
    return max(0, min(100, score))


def calculate_functional_score(row):
    """Functional scoring based on bike specifications - NOW INCLUDES FUEL EFFICIENCY"""
    score = 0
    
    # Engine Performance (40 points) - REDUCED from 45
    score += row.get('Engine Displacement_Norm', 0) * 0.13
    score += row.get('Max power PS_Norm', 0) * 0.16
    score += row.get('Max Torque By Nm_Norm', 0) * 0.11
    
    # Fuel Efficiency (5 points) - NEWLY ADDED
    score += row.get('Fuel efficiency_Norm', 0) * 0.05
    
    # RPM characteristics (10 points)
    score += row.get('Max power RPM_Norm', 0) * 0.05
    score += row.get('Max Torque RPM_Norm', 0) * 0.05
    
    # Chassis & Handling (30 points)
    score += row.get('Ground Clearance mm_Norm', 0) * 0.10
    score += row.get('Wheel Base mm_Norm', 0) * 0.10
    score += row.get('Fuel Tank Litre_Norm', 0) * 0.10
    
    # Tire Quality (15 points)
    front_tire = row.get('Front Tyres Size width in mm_Norm', 0) * 0.075
    rear_tire = row.get('Rear Tyres Size width in mm_Norm', 0) * 0.075
    score += front_tire + rear_tire
    
    return max(0, min(100, score))


def calculate_safety_score(row):
    """Safety scoring based on objective safety features"""
    score = 0
    
    # ABS System (50 points)
    abs_type = str(row.get('ABS', 'No')).strip().lower()
    if 'dual' in abs_type or abs_type == 'dual channel':
        score += 50
    elif 'single' in abs_type or abs_type == 'single channel':
        score += 30
    else:
        score += 0
    
    # Front Brake (25 points)
    front_brake = str(row.get('Front Brake Type', '')).strip().lower()
    if 'disc' in front_brake:
        score += 25
    else:
        score += 10
    
    # Rear Brake (25 points)
    rear_brake = str(row.get('Rear Brake Type', '')).strip().lower()
    if 'disc' in rear_brake:
        score += 25
    else:
        score += 15
    
    return max(0, min(100, score))


def calculate_terrain_purpose_score(row, selected_terrains, selected_purposes):
    """Score based on terrain and purpose match"""
    if not selected_terrains and not selected_purposes:
        return 50  # Neutral score if no preference
    
    bike_type = str(row.get('Bike Type', '')).strip()
    score = 0
    max_possible = 0
    
    # Terrain matching (50 points max)
    if selected_terrains:
        max_possible += 50
        terrain_matches = 0
        for terrain in selected_terrains:
            if bike_type in TERRAIN_BIKE_MAP[terrain]['types']:
                terrain_matches += 1
        if terrain_matches > 0:
            score += (terrain_matches / len(selected_terrains)) * 50
    
    # Purpose matching (50 points max)
    if selected_purposes:
        max_possible += 50
        purpose_matches = 0
        for purpose in selected_purposes:
            if bike_type in PURPOSE_BIKE_MAP[purpose]['types']:
                purpose_matches += 1
        if purpose_matches > 0:
            score += (purpose_matches / len(selected_purposes)) * 50
    
    # Normalize to 0-100
    if max_possible > 0:
        score = (score / max_possible) * 100
    else:
        score = 50
    
    return max(0, min(100, score))


def calculate_value_score(row, budget_range):
    """Value scoring - rewards bikes in sweet spot of budget"""
    price = row.get('Price (Rs)', 0)
    budget_min, budget_max = budget_range
    
    if price < budget_min or price > budget_max:
        return 0
    
    budget_span = budget_max - budget_min
    if budget_span == 0:
        return 100
    
    optimal_low = budget_min + (budget_span * 0.30)
    optimal_high = budget_min + (budget_span * 0.60)
    
    if optimal_low <= price <= optimal_high:
        return 100
    elif price < optimal_low:
        ratio = (price - budget_min) / (optimal_low - budget_min)
        return 70 + (ratio * 30)
    else:
        ratio = (price - optimal_high) / (budget_max - optimal_high)
        return 100 - (ratio * 40)


# ==================== CALCULATE SCORES ==================== #
if not df_filtered.empty:
    df_filtered['Ergonomic Score'] = df_filtered.apply(
        lambda row: calculate_ergonomic_score(row, rider_height_mm, rider_weight, inseam_mm), 
        axis=1
    )
    
    df_filtered['Functional Score'] = df_filtered.apply(calculate_functional_score, axis=1)
    
    df_filtered['Safety Score'] = df_filtered.apply(calculate_safety_score, axis=1)
    
    df_filtered['Terrain/Purpose Score'] = df_filtered.apply(
        lambda row: calculate_terrain_purpose_score(row, selected_terrains, selected_purposes),
        axis=1
    )
    
    df_filtered['Value Score'] = df_filtered.apply(
        lambda row: calculate_value_score(row, budget), 
        axis=1
    )
    
    # TOTAL SCORE
    df_filtered['Total Score'] = (
        df_filtered['Ergonomic Score'] * 0.30 +
        df_filtered['Functional Score'] * 0.30 +
        df_filtered['Safety Score'] * 0.20 +
        df_filtered['Terrain/Purpose Score'] * 0.10 +
        df_filtered['Value Score'] * 0.10
    )
    
    # Sort by total score
    df_filtered = df_filtered.sort_values('Total Score', ascending=False)

# ==================== RESULTS DISPLAY ==================== #
st.markdown("---")
st.markdown("## 🏆 Top Recommendations")

if df_filtered.empty:
    st.error("❌ No bikes match your filters. Try widening your criteria.")
else:
    # Top 15 recommendations
    top_15 = df_filtered.head(15)
    
    # Display ALL columns - scores first, then all bike specs
    score_cols = ['Total Score', 'Ergonomic Score', 'Functional Score', 
                  'Safety Score', 'Terrain/Purpose Score', 'Value Score']
    
    # Get all original columns except normalized ones
    original_cols = [col for col in top_15.columns if not col.endswith('_Norm')]
    
    # Remove score columns from original list to avoid duplication
    spec_cols = [col for col in original_cols if col not in score_cols]
    
    # Reorder: Bike Names first, then Brand, Type, Price, then Scores, then all other specs
    priority_cols = ['Bike Names', 'Brand', 'Bike Type', 'Price (Rs)']
    display_cols = []
    
    # Add priority columns that exist
    for col in priority_cols:
        if col in spec_cols:
            display_cols.append(col)
            spec_cols.remove(col)
    
    # Add score columns
    display_cols.extend([col for col in score_cols if col in top_15.columns])
    
    # Add all remaining specification columns
    display_cols.extend(spec_cols)
    
    # Create format dictionary for all numeric columns
    format_dict = {}
    for col in display_cols:
        if col in top_15.columns:
            if 'Price' in col:
                format_dict[col] = '{:,.0f}'
            elif 'Score' in col:
                format_dict[col] = '{:.1f}'
            elif top_15[col].dtype in ['float64', 'float32']:
                format_dict[col] = '{:.1f}'
            elif top_15[col].dtype in ['int64', 'int32']:
                format_dict[col] = '{:.0f}'
    
    st.dataframe(
        top_15[display_cols].style.format(format_dict).background_gradient(subset=['Total Score'], cmap='Greens'),
        width='stretch',
        height=500
    )
    
    # Download option
    csv = top_15[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Top 15 Recommendations (CSV)",
        data=csv,
        file_name="bike_recommendations.csv",
        mime="text/csv"
    )
    
    # ==================== VISUALIZATIONS ==================== #
    st.markdown("---")
    st.markdown("## 📊 Visual Analysis")
    
    # Top 10 Total Score Bar Chart
    top_10 = df_filtered.head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.barh(range(len(top_10)), top_10['Total Score'], color='steelblue')
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['Bike Names'], fontsize=9)
    ax1.set_xlabel('Total Score', fontsize=11)
    ax1.set_title('Top 10 Bikes by Total Score', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    for i, (idx, row) in enumerate(top_10.iterrows()):
        ax1.text(row['Total Score'] + 1, i, f"{row['Total Score']:.1f}", 
                va='center', fontsize=9)
    
    st.pyplot(fig1)
    
    # Score Components Comparison
    st.markdown("### Score Component Breakdown (Top 5)")
    top_5 = df_filtered.head(5)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_5))
    width = 0.16
    
    ax2.bar(x - 2*width, top_5['Ergonomic Score'], width, label='Ergonomic (30%)', color='#2E86AB')
    ax2.bar(x - width, top_5['Functional Score'], width, label='Functional (30%)', color='#A23B72')
    ax2.bar(x, top_5['Safety Score'], width, label='Safety (20%)', color='#F18F01')
    ax2.bar(x + width, top_5['Terrain/Purpose Score'], width, label='Terrain/Purpose (10%)', color='#06A77D')
    ax2.bar(x + 2*width, top_5['Value Score'], width, label='Value (10%)', color='#C73E1D')
    
    ax2.set_xlabel('Bikes', fontsize=11)
    ax2.set_ylabel('Score (0-100)', fontsize=11)
    ax2.set_title('Score Components for Top 5 Recommendations', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_5['Bike Names'], rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Ergonomic Fit Analysis
    st.markdown("### Ergonomic Fit Analysis (Top 10)")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Seat Height vs Inseam
    ax3a.scatter(top_10['Seat Height mm'], top_10['Ergonomic Score'], 
                s=100, alpha=0.6, c=top_10['Ergonomic Score'], cmap='RdYlGn')
    ax3a.axvline(inseam_mm, color='red', linestyle='--', linewidth=2, label=f'Your Inseam ({inseam_mm:.0f}mm)')
    ax3a.axvline(inseam_mm * 0.9, color='orange', linestyle='--', linewidth=2, label=f'Ideal Max ({inseam_mm*0.9:.0f}mm)')
    ax3a.set_xlabel('Seat Height (mm)', fontsize=11)
    ax3a.set_ylabel('Ergonomic Score', fontsize=11)
    ax3a.set_title('Seat Height vs Ergonomic Score', fontsize=12, fontweight='bold')
    ax3a.legend(fontsize=9)
    ax3a.grid(alpha=0.3)
    
    # Weight Ratio Analysis
    top_10_copy = top_10.copy()
    top_10_copy['Weight Ratio'] = top_10_copy['Kerb Weight'] / rider_weight
    ax3b.scatter(top_10_copy['Weight Ratio'], top_10_copy['Ergonomic Score'], 
                s=100, alpha=0.6, c=top_10_copy['Ergonomic Score'], cmap='RdYlGn')
    ax3b.axvline(2.0, color='green', linestyle='--', linewidth=2, label='Easy (≤2.0x)')
    ax3b.axvline(2.5, color='orange', linestyle='--', linewidth=2, label='Moderate (≤2.5x)')
    ax3b.axvline(3.0, color='red', linestyle='--', linewidth=2, label='Challenging (≤3.0x)')
    ax3b.set_xlabel('Bike Weight / Rider Weight Ratio', fontsize=11)
    ax3b.set_ylabel('Ergonomic Score', fontsize=11)
    ax3b.set_title('Weight Ratio vs Ergonomic Score', fontsize=12, fontweight='bold')
    ax3b.legend(fontsize=9)
    ax3b.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Terrain/Purpose Match Distribution
    if selected_terrains or selected_purposes:
        st.markdown("### Terrain/Purpose Match Distribution (Top 10)")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        bike_type_scores = top_10.groupby('Bike Type')['Terrain/Purpose Score'].mean().sort_values(ascending=False)
        
        bars = ax4.barh(range(len(bike_type_scores)), bike_type_scores.values, color='forestgreen')
        ax4.set_yticks(range(len(bike_type_scores)))
        ax4.set_yticklabels(bike_type_scores.index, fontsize=10)
        ax4.set_xlabel('Average Terrain/Purpose Match Score', fontsize=11)
        ax4.set_title('How Well Bike Types Match Your Preferences', fontsize=13, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)
        
        for i, val in enumerate(bike_type_scores.values):
            ax4.text(val + 1, i, f"{val:.1f}", va='center', fontsize=9)
        
        st.pyplot(fig4)
    
    # Specification Scatter Plots
    st.markdown("### Specification Analysis (All Filtered Bikes)")
    
    spec_col1, spec_col2 = st.columns(2)
    
    with spec_col1:
        fig5a, ax5a = plt.subplots(figsize=(7, 5))
        scatter = ax5a.scatter(df_filtered['Max power PS'], 
                              df_filtered['Max Torque By Nm'],
                              c=df_filtered['Total Score'], 
                              cmap='viridis', 
                              s=60, alpha=0.6)
        ax5a.set_xlabel('Max Power (PS)', fontsize=11)
        ax5a.set_ylabel('Max Torque (Nm)', fontsize=11)
        ax5a.set_title('Power vs Torque (colored by Total Score)', fontsize=12, fontweight='bold')
        ax5a.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax5a, label='Total Score')
        st.pyplot(fig5a)
    
    with spec_col2:
        fig5b, ax5b = plt.subplots(figsize=(7, 5))
        scatter = ax5b.scatter(df_filtered['Kerb Weight'], 
                              df_filtered['Price (Rs)'],
                              c=df_filtered['Total Score'], 
                              cmap='viridis', 
                              s=60, alpha=0.6)
        ax5b.set_xlabel('Kerb Weight (kg)', fontsize=11)
        ax5b.set_ylabel('Price (Rs)', fontsize=11)
        ax5b.set_title('Weight vs Price (colored by Total Score)', fontsize=12, fontweight='bold')
        ax5b.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax5b, label='Total Score')
        st.pyplot(fig5b)
    
    # Ground Clearance vs Fuel Efficiency
    if 'Ground Clearance mm' in df_filtered.columns and 'Fuel efficiency' in df_filtered.columns:
        spec_col3, spec_col4 = st.columns(2)
        
        with spec_col3:
            fig5c, ax5c = plt.subplots(figsize=(7, 5))
            scatter = ax5c.scatter(df_filtered['Ground Clearance mm'], 
                                  df_filtered['Fuel efficiency'],
                                  c=df_filtered['Terrain/Purpose Score'], 
                                  cmap='plasma', 
                                  s=60, alpha=0.6)
            ax5c.set_xlabel('Ground Clearance (mm)', fontsize=11)
            ax5c.set_ylabel('Fuel Efficiency (kmpl)', fontsize=11)
            ax5c.set_title('Ground Clearance vs Fuel Efficiency', fontsize=12, fontweight='bold')
            ax5c.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax5c, label='Terrain/Purpose Score')
            st.pyplot(fig5c)
        
        with spec_col4:
            fig5d, ax5d = plt.subplots(figsize=(7, 5))
            scatter = ax5d.scatter(df_filtered['Engine Displacement'], 
                                  df_filtered['Fuel efficiency'],
                                  c=df_filtered['Functional Score'], 
                                  cmap='coolwarm', 
                                  s=60, alpha=0.6)
            ax5d.set_xlabel('Engine Displacement (cc)', fontsize=11)
            ax5d.set_ylabel('Fuel Efficiency (kmpl)', fontsize=11)
            ax5d.set_title('Displacement vs Fuel Efficiency', fontsize=12, fontweight='bold')
            ax5d.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax5d, label='Functional Score')
            st.pyplot(fig5d)
    
    # ==================== DETAILED COMPARISON ==================== #
    st.markdown("---")
    st.markdown("## 🔍 Detailed Bike Comparison")
    
    comparison_bikes = st.multiselect(
        "Select 2-5 bikes to compare in detail",
        options=df_filtered['Bike Names'].tolist(),
        default=df_filtered['Bike Names'].head(min(2, len(df_filtered))).tolist()
    )
    
    if len(comparison_bikes) >= 2:
        comparison_df = df_filtered[df_filtered['Bike Names'].isin(comparison_bikes)]
        
        # Radar Chart
        if len(comparison_bikes) <= 5:
            st.markdown("### Multi-Dimensional Comparison (Radar Chart)")
            
            categories = ['Ergonomic', 'Functional', 'Safety', 'Terrain/Purpose', 'Value']
            
            fig6, ax6 = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#C73E1D']
            
            for idx, (_, bike) in enumerate(comparison_df.iterrows()):
                values = [bike[cat + ' Score'] for cat in categories]
                values += values[:1]
                
                color = colors[idx % len(colors)]
                ax6.plot(angles, values, 'o-', linewidth=2, label=bike['Bike Names'], color=color)
                ax6.fill(angles, values, alpha=0.15, color=color)
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(categories, fontsize=11)
            ax6.set_ylim(0, 100)
            ax6.set_title('Multi-dimensional Bike Comparison', fontsize=14, fontweight='bold', pad=20)
            ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
            ax6.grid(True)
            
            st.pyplot(fig6)
        
        # Specifications Table
        st.markdown("### Detailed Specifications Comparison")
        comparison_cols = ['Bike Names', 'Brand', 'Price (Rs)', 'Bike Type',
                          'Engine Displacement', 'Max power PS', 'Max Torque By Nm',
                          'Seat Height mm', 'Kerb Weight', 'Ground Clearance mm',
                          'Fuel Tank Litre', 'Fuel efficiency', 'ABS', 
                          'Front Brake Type', 'Rear Brake Type',
                          'Total Score', 'Ergonomic Score', 'Functional Score', 
                          'Safety Score', 'Terrain/Purpose Score', 'Value Score']
        
        comparison_cols = [col for col in comparison_cols if col in comparison_df.columns]
        
        st.dataframe(
            comparison_df[comparison_cols].style.format({
                'Price (Rs)': '{:,.0f}',
                'Total Score': '{:.1f}',
                'Ergonomic Score': '{:.1f}',
                'Functional Score': '{:.1f}',
                'Safety Score': '{:.1f}',
                'Terrain/Purpose Score': '{:.1f}',
                'Value Score': '{:.1f}',
                'Engine Displacement': '{:.0f}',
                'Max power PS': '{:.1f}',
                'Max Torque By Nm': '{:.1f}',
                'Seat Height mm': '{:.0f}',
                'Kerb Weight': '{:.0f}',
                'Ground Clearance mm': '{:.0f}',
                'Fuel Tank Litre': '{:.1f}',
                'Fuel efficiency': '{:.1f}'
            }),
            width='stretch'
        )
        
        # Side-by-side bar comparison
        st.markdown("### Score Comparison (Bar Chart)")
        fig7, ax7 = plt.subplots(figsize=(12, 6))
        
        score_categories = ['Ergonomic Score', 'Functional Score', 'Safety Score', 
                           'Terrain/Purpose Score', 'Value Score']
        x = np.arange(len(score_categories))
        width = 0.8 / len(comparison_bikes)
        
        for idx, (_, bike) in enumerate(comparison_df.iterrows()):
            offset = (idx - len(comparison_bikes)/2) * width + width/2
            values = [bike[cat] for cat in score_categories]
            ax7.bar(x + offset, values, width, label=bike['Bike Names'])
        
        ax7.set_xlabel('Score Components', fontsize=11)
        ax7.set_ylabel('Score (0-100)', fontsize=11)
        ax7.set_title('Score Component Comparison', fontsize=13, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(['Ergonomic', 'Functional', 'Safety', 'Terrain/Purpose', 'Value'], fontsize=10)
        ax7.legend(fontsize=9)
        ax7.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7)
    
    # ==================== INSIGHTS & RECOMMENDATIONS ==================== #
    st.markdown("---")
    st.markdown("## 💡 Personalized Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("### Best Ergonomic Fit")
        best_ergo = df_filtered.nlargest(3, 'Ergonomic Score')[['Bike Names', 'Ergonomic Score', 'Seat Height mm', 'Kerb Weight']]
        st.dataframe(best_ergo.style.format({
            'Ergonomic Score': '{:.1f}',
            'Seat Height mm': '{:.0f}',
            'Kerb Weight': '{:.0f}'
        }), width='stretch')
        
        st.markdown("### Best Value for Money")
        best_value = df_filtered.nlargest(3, 'Value Score')[['Bike Names', 'Value Score', 'Price (Rs)']]
        st.dataframe(best_value.style.format({
            'Value Score': '{:.1f}',
            'Price (Rs)': '{:,.0f}'
        }), width='stretch')
    
    with insight_col2:
        st.markdown("### Best Performance")
        best_func = df_filtered.nlargest(3, 'Functional Score')[['Bike Names', 'Functional Score', 'Max power PS', 'Max Torque By Nm']]
        st.dataframe(best_func.style.format({
            'Functional Score': '{:.1f}',
            'Max power PS': '{:.1f}',
            'Max Torque By Nm': '{:.1f}'
        }), width='stretch')
        
        st.markdown("### Safest Options")
        best_safety = df_filtered.nlargest(3, 'Safety Score')[['Bike Names', 'Safety Score', 'ABS']]
        st.dataframe(best_safety.style.format({
            'Safety Score': '{:.1f}'
        }), width='stretch')
    
    # Key Statistics
    st.markdown("### Your Profile Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        avg_ergo = df_filtered['Ergonomic Score'].mean()
        st.metric("Avg Ergonomic Match", f"{avg_ergo:.1f}/100")
    
    with summary_col2:
        perfect_fit = len(df_filtered[df_filtered['Seat Height mm'] <= inseam_mm * 0.9])
        st.metric("Perfect Seat Height Fits", f"{perfect_fit} bikes")
    
    with summary_col3:
        manageable_weight = len(df_filtered[df_filtered['Kerb Weight'] / rider_weight <= 2.5])
        st.metric("Manageable Weight Bikes", f"{manageable_weight} bikes")
    
    with summary_col4:
        dual_abs = len(df_filtered[df_filtered['ABS'].str.contains('Dual', case=False, na=False)])
        st.metric("Bikes with Dual ABS", f"{dual_abs} bikes")
    
    # Recommendations based on profile
    st.markdown("---")
    st.markdown("### 📋 Personalized Recommendations")
    
    if inseam_mm < 750:
        st.info("🔹 **Low Seat Height Priority**: Your inseam suggests focusing on bikes with seat height ≤ 750mm for better ground reach.")
    
    if rider_weight < 55:
        st.info("🔹 **Lightweight Bikes Recommended**: Consider bikes under 140kg for easier handling at stops.")
    
    if selected_terrains and 'Mountain/Off-road' in selected_terrains:
        high_gc = df_filtered[df_filtered['Ground Clearance mm'] >= 180]
        st.success(f"🔹 **Off-road Ready**: {len(high_gc)} bikes have ground clearance ≥ 180mm suitable for mountain terrain.")
    
    if selected_purposes and 'Long Distance Touring' in selected_purposes:
        good_touring = df_filtered[(df_filtered['Fuel Tank Litre'] >= 12) & (df_filtered['Seat Height mm'] <= inseam_mm)]
        st.success(f"🔹 **Touring Capable**: {len(good_touring)} bikes combine large fuel tanks with comfortable ergonomics for long rides.")

# ==================== FOOTER ==================== #
st.markdown("---")
st.markdown("""
### 📚 Research Context & Updates
This system addresses:
- **RQ1**: Specifications influence through weighted functional scoring
- **RQ2**: Rider characteristics impact via ergonomic matching (height→seat, weight→handling)
- **H1**: Spec-based matching vs brand preference (use without brand filter to test)
- **H2**: Reducing bias by prioritizing measurable specs over marketing factors

**🆕 Version  Updates:**
- ✅ **Fuel efficiency now integrated into Functional Score (5% weight)**
- Rebalanced engine performance from 45% to 40% to accommodate fuel efficiency
- All available data columns now utilized in scoring system
- Improved real-world relevance for budget-conscious young riders

**Next Steps for Validation:**
1. Survey users: satisfaction with recommendations vs traditional choice
2. Compare recommended bikes to social media popular bikes
3. Track actual purchases and long-term satisfaction
4. Validate weight ratio thresholds with rider interviews

*Dashboard Version  - Complete Data Utilization with Fuel Efficiency Integration*
""")