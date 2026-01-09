import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== CONFIGURATION ==================== #
st.set_page_config(
    layout='wide', 
    page_title="Multi-Criteria Motorcycle Decision Support - Nepal",
    page_icon="🏍️"
)

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
    df.columns = df.columns.str.strip()  # CRITICAL: Remove trailing spaces (e.g., 'ABS ')
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().str.replace('"','').str.replace("'",'')
    
    # Handle Kerb Weight column naming
    if 'Kerb Weight mm' in df.columns:
        df.rename(columns={'Kerb Weight mm':'Kerb Weight'}, inplace=True)
    elif 'Kerb Weight kg' in df.columns:
        df.rename(columns={'Kerb Weight kg':'Kerb Weight'}, inplace=True)
    
    # CRITICAL: Rename Fuel efficiency columns
    if 'Fuel efficiency mileage' in df.columns:
        df.rename(columns={'Fuel efficiency mileage':'Fuel efficiency'}, inplace=True)
    
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
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True), errors='coerce').fillna(0)
    
    return df

# Try to load bike_cleaned.xlsx first, fallback to top_bikes.csv if not found
file_path = "bike_cleaned.xlsx"
if not os.path.exists(file_path):
    file_path = "top_bikes.csv"
    
df = load_data(file_path)

# THESIS REQUIREMENT: Filter bikes below 4 lakhs (400,000 Rs)
# Urban youth (18-25) in Nepal - realistic purchasing power
df = df[df['Price (Rs)'] >= 400000].copy()

if df.empty:
    st.error("⚠️ No motorcycles found above ₹400,000 (4 lakhs). Please check your dataset.")
    st.info("💡 This framework is designed for urban youth (18-25) with purchasing power of 4 lakhs and above.")
    st.stop()

# ==================== TITLE AND METHODOLOGY ==================== #
st.title("🏍️ Multi-Criteria Decision Support Framework for Motorcycle Selection")
st.markdown("### Specification-Based Analysis with Behavioral Prioritization for Urban Youth (18-25) in Nepal")

with st.expander("📊 **Research Methodology & System Logic**"):
    st.markdown("""
    **Thesis Title:** Design and Development of a Multi-Criteria Decision Support Framework for Motorcycle Selection Among Urban Youth (18–25) in Nepal Using Specification-Based Analysis and Behavioral Prioritization
    
    **Research Questions:**
    - RQ1: How do behavioral economics theories (herd behavior, cognitive biases) explain suboptimal motorcycle selection among urban Nepali youth, and how can a multi-criteria decision support framework mitigate these biases?
    - RQ2: What are the ethical and cultural implications of deploying a decision support framework that challenges popularity-driven and status-oriented motorcycle choices?
    
    **Hypotheses:**
    - H1: Urban youth following popularity-driven trends (herd behavior) select motorcycles with lower functional suitability, safety, or economic value compared to choices informed by structured multi-criteria frameworks
    - H2: Decision support frameworks countering social conformity may face ethical/cultural resistance unless they incorporate mechanisms for cultural acceptance and transparency
    
    **Scoring System Components:**
    
    **1. Ergonomic Score (30%)** - Rider-bike physical compatibility
    - Seat Height Fit (60%): seat height vs inseam (ideal: ≤90% of inseam)
    - Weight Handling (40%): bike/rider weight ratio for control at stops
    - BMI calculated for reference but not scored (acknowledged limitation)
    
    **2. Functional Score (35%)** - Performance specifications ⭐ INCREASED FOR THESIS
    - Engine: displacement, power, torque (40%)
    - Fuel Economy (5%): fuel efficiency for cost-effectiveness
    - Chassis: ground clearance, wheelbase, fuel capacity (30%)
    - Tires: width and aspect ratio (15%)
    - RPM characteristics (10%)
    
    **3. Safety Score (20%)** - Objective safety features
    - ABS type: Dual/Single/None (50%)
    - Brake types: Disc/Drum front and rear (50%)
    
    **4. Terrain/Purpose Match Score (10%)** - Use-case suitability ⭐ SPEC-BASED (NOT TYPE-BASED)
    - CRITICAL CHANGE: Now based on FUNCTIONAL SPECS instead of bike type classification
    - For Off-road/Mountain: Ground Clearance + Torque + Weight
    - For Urban: Seat Height + Weight + Fuel Efficiency
    - For Touring: Fuel Tank + Comfort + Power
    - **Why this matters for Nepali riders**: A light commuter may be classified "Commuter" but have high torque and can handle rough roads better than a "Naked Sport"
    
    **5. Value Score (5%)** - Price optimization ⭐ REDUCED FOR THESIS
    - Rewards bikes in 30-60% budget range (sweet spot)
    - Lowered weight because functional fit > price for thesis focus
    
    **Addressing Behavioral Biases (Thesis Focus):**
    - **Herd Behavior Mitigation**: Brand filter available but flagged as potential source of popularity-driven bias
    - **Specification-Based vs Type-Based**: Terrain/purpose matching uses ACTUAL SPECS (GC, torque, weight) instead of bike type labels to counter marketing categorization
    - **Transparency**: System makes criteria weights explicit (Ergonomic 30%, Functional 35%, Safety 20%, Terrain 10%, Value 5%)
    - **Behavioral Prioritization**: Users can adjust filters to reflect their priorities while seeing objective specifications
    - **Social Influence Awareness**: Framework highlights when popular choices may not match functional needs
    
    **Why This Matters for Urban Nepali Youth (18-25):**
    - **Bounded Rationality**: Young riders face information overload; structured framework reduces cognitive burden
    - **Status vs Function**: Framework makes trade-offs visible between social signaling (brand, power) and practical needs (fuel efficiency, maintenance)
    - **Economic Constraints**: Value scoring helps identify sweet spots in budget range
    - **Safety Considerations**: Explicit safety scoring (ABS, brakes) counters tendency to prioritize style over safety
    - **Nepali Context**: Diverse terrain (Kathmandu valley, hills, mountains) requires specification-based matching, not just brand popularity
    
    **Theoretical Foundation:**
    - **Decision Theory**: Multi-criteria decision analysis (MCDA) structures complex trade-offs
    - **Behavioral Economics**: Addresses herd behavior, loss aversion, anchoring bias, status-driven consumption
    - **Bounded Rationality (Simon)**: Framework extends cognitive capacity without assuming perfect rationality
    - **Prospect Theory (Kahneman & Tversky)**: Recognizes that youth may avoid socially "risky" choices even if functionally superior
    
    **Acknowledged Limitations:**
    - Weight ratios use simplified thresholds (need empirical validation with Nepali riders)
    - Missing ergonomic data: handlebar reach, knee angles, footpeg position, riding position lean
    - Normalization based on dataset range, not universal engineering standards
    - Terrain thresholds based on typical characteristics (need field validation on Nepali roads)
    - Does NOT account for rider experience level (assumes first-time or early buyers aged 18-25)
    - Framework is conceptual/theoretical - not empirically validated through user studies
    """)


# =========== DATASET SUMMARY ====== #
st.markdown("---")
st.markdown("## 📈 Dataset Overview")
st.info("🎓 **Thesis Scope**: Dataset filtered to motorcycles ≥ ₹400,000 (4 lakhs) - aligned with urban youth (18-25) purchasing power in Nepal")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Bikes (≥4L)", len(df))
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

# ========= USER INPUTS ====== #
st.markdown("---")
st.markdown("## 👤 Rider Physical Characteristics")
st.info("📋 **Thesis Context**: Ergonomic fit is critical for urban youth (18-25) who may be first-time buyers. Physical mismatch can lead to safety risks and ownership dissatisfaction.")

col_left, col_mid, col_right = st.columns([1,1,1])

with col_left:
    rider_height_cm = st.number_input("Height (cm)", 140, 210, 170, help="Your height in centimeters")
    rider_height_mm = rider_height_cm * 10  # Convert to mm for calculations
    
with col_mid:
    rider_weight = st.number_input("Weight (kg)", 40, 120, 65, help="Your weight in kilograms")
    
with col_right:
    bmi = rider_weight / ((rider_height_cm/100)**2)
    st.metric("Calculated BMI", f"{bmi:.2f}")

# Calculate inseam
inseam_mm = rider_height_mm * 0.45
inseam_cm = inseam_mm / 10

# ===== BMI-BASED RIDER PROFILING ===== #
if rider_height_cm < 160:
    height_type = "SHORT"
elif rider_height_cm > 185:
    height_type = "TALL"
else:
    height_type = "AVERAGE"

if bmi < 20:
    bmi_type = "Lean"
elif bmi < 25:
    bmi_type = "Athletic"
else:
    bmi_type = "Heavy"

rider_profile = f"{height_type}_{bmi_type}"

with st.expander("👤 Your Physical Profile (RQ2: Height, Weight, BMI Interaction)", expanded=False):
    prof_col1, prof_col2, prof_col3 = st.columns(3)
    with prof_col1:
        st.metric("Height Category", height_type)
    with prof_col2:
        st.metric("BMI Type", bmi_type)
    with prof_col3:
        st.metric("Rider Profile", rider_profile)
    
    profile_insights = {
        "SHORT_Lean": {
            "ergonomic": "Very low seat height critical (<750mm)",
            "specs": "Seat height, light weight, stability"
        },
        "SHORT_Athletic": {
            "ergonomic": "Low seat height needed (700-770mm)",
            "specs": "Seat height, balance, power delivery"
        },
        "SHORT_Heavy": {
            "ergonomic": "LOW seat height CRITICAL (<730mm)",
            "specs": "Seat height (most critical), weight ratio, power"
        },
        "AVERAGE_Lean": {
            "ergonomic": "Wide range (750-820mm) - flexible",
            "specs": "Power, handling, torque"
        },
        "AVERAGE_Athletic": {
            "ergonomic": "Standard range (770-810mm)",
            "specs": "Balanced specs, comfort, performance"
        },
        "AVERAGE_Heavy": {
            "ergonomic": "Moderate to high seat (780-850mm)",
            "specs": "Weight ratio, seat height, power"
        },
        "TALL_Lean": {
            "ergonomic": "High seat height (830-880mm)",
            "specs": "Seat height, handlebar reach, knee angle"
        },
        "TALL_Athletic": {
            "ergonomic": "High seat height (820-860mm)",
            "specs": "Seat height, wheelbase, comfort"
        },
        "TALL_Heavy": {
            "ergonomic": "Very high seat (850mm+) - challenging",
            "specs": "Seat height, weight ratio, power"
        }
    }
    
    if rider_profile in profile_insights:
        insight = profile_insights[rider_profile]
        st.write(f"**Ergonomic Fit Priority**: {insight['ergonomic']}")
        st.write(f"**Focus Specs**: {insight['specs']}")

st.info(f"📏 **Estimated Inseam: {inseam_cm:.1f} cm ({inseam_mm:.0f} mm)** - Ideal seat height: ≤ {inseam_cm * 0.9:.1f} cm for flat-foot reach")

# =========== RIDING CONTEXT ======== #
st.markdown("---")
st.markdown("## 🎯 Riding Purpose & Terrain")
st.info("🎯 **Thesis Objective**: Understanding how specifications match terrain/purpose needs helps counter popularity-driven choices. Nepal's diverse geography (urban Kathmandu, mountain passes, rural roads) requires specification-based evaluation.")

terrain_purpose_col1, terrain_purpose_col2 = st.columns(2)

# Terrain mapping based on bike specifications
TERRAIN_BIKE_MAP = {
    "Urban/City": {
        "types": ["Commuter", "Scooter", "Naked Sport", "Streetfighter"],
        "specs": "Low seat height, light weight, fuel efficient, easy maneuverability",
        "why": "Dense traffic, potholes, narrow roads"
    },
    "Highway/Touring": {
        "types": ["Sport", "Fully-faired Sport", "Sport-touring", "Cruiser", "Roadster"],
        "specs": "Higher power, larger fuel tank, comfortable ergonomics, windscreen",
        "why": "Long distances, highway speed stability, comfort on extended rides"
    },
    "Mountain/Off-road": {
        "types": ["Adventure", "Off-road", "Dual-sport", "Scrambler", "Enduro"],
        "specs": "High ground clearance, long suspension travel, knobby tires, light weight",
        "why": "Steep climbs, rough terrain, hairpins, poor road quality"
    },
    "Mixed/Versatile": {
        "types": ["Adventure", "Dual-sport", "Streetfighter", "Naked Sport"],
        "specs": "Balanced specs, moderate seat height, good ground clearance",
        "why": "Variety of riding conditions - city, mountains, rural"
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

# =========== BUDGET & CORE FILTERS ========= #
st.markdown("---")
st.markdown("## 💰 Budget & Core Specifications")

col1, col2 = st.columns(2)
with col1:
    # Budget range - dataset already filtered to 4 lakhs minimum
    budget_min = int(df['Price (Rs)'].min())
    budget_max = int(df['Price (Rs)'].max())
    default_min = budget_min
    default_max = budget_max  # Show ALL bikes above 4 lakhs by default
    budget = st.slider("Budget Range (Rs)", budget_min, budget_max, (default_min, default_max), step=10000)
    st.caption(f"💡 Dataset pre-filtered: All bikes ≥ ₹400,000 (4 lakhs) - Adjust slider to narrow your budget range")

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
                            float(df['Max power PS'].min()), 
                            float(df['Max power PS'].max()), 
                            (float(df['Max power PS'].min()), float(df['Max power PS'].max())),
                            step=0.5)
with perf_col2:
    torque_range = st.slider("Torque (Nm)", 
                             float(df['Max Torque By Nm'].min()), 
                             float(df['Max Torque By Nm'].max()), 
                             (float(df['Max Torque By Nm'].min()), float(df['Max Torque By Nm'].max())),
                             step=0.5)
with perf_col3:
    displacement_range = st.slider("Engine Displacement (cc)", 
                                    float(df['Engine Displacement'].min()), 
                                    float(df['Engine Displacement'].max()), 
                                    (float(df['Engine Displacement'].min()), float(df['Engine Displacement'].max())),
                                    step=5.0)

# ============= ADDITIONAL SPEC FILTERS (ENHANCED) ========= #
st.markdown("### Additional Specification Filters")
add_spec_col1, add_spec_col2, add_spec_col3, add_spec_col4 = st.columns(4)

with add_spec_col1:
    # NEW: Seat Height Filter
    if 'Seat Height mm' in df.columns:
        use_seat_filter = st.checkbox("Filter Seat Height", value=False, 
                                      help="Critical for ergonomic fit")
        if use_seat_filter:
            seat_range = st.slider("Seat Height (mm)",
                                int(df['Seat Height mm'].min()),
                                int(df['Seat Height mm'].max()),
                                (int(df['Seat Height mm'].min()), int(inseam_mm)))
        else:
            seat_range = None
    else:
        seat_range = None

with add_spec_col2:
    if 'Ground Clearance mm' in df.columns:
        use_gc_filter = st.checkbox("Filter Ground Clearance", value=False)
        if use_gc_filter:
            gc_range = st.slider("Ground Clearance (mm)",
                                float(df['Ground Clearance mm'].min()),
                                float(df['Ground Clearance mm'].max()),
                                (float(df['Ground Clearance mm'].min()), float(df['Ground Clearance mm'].max())),
                                step=1.0)
        else:
            gc_range = None
    else:
        gc_range = None

with add_spec_col3:
    if 'Fuel Tank Litre' in df.columns:
        use_fuel_filter = st.checkbox("Filter Fuel Tank", value=False)
        if use_fuel_filter:
            fuel_range = st.slider("Fuel Tank Capacity (L)",
                                  float(df['Fuel Tank Litre'].min()),
                                  float(df['Fuel Tank Litre'].max()),
                                  (float(df['Fuel Tank Litre'].min()), float(df['Fuel Tank Litre'].max())),
                                  step=0.1)
        else:
            fuel_range = None
    else:
        fuel_range = None

with add_spec_col4:
    if 'Fuel efficiency' in df.columns:
        use_fe_filter = st.checkbox("Filter Fuel Efficiency", value=False)
        if use_fe_filter:
            fe_range = st.slider("Fuel Efficiency (kmpl)",
                                float(df['Fuel efficiency'].min()),
                                float(df['Fuel efficiency'].max()),
                                (float(df['Fuel efficiency'].min()), float(df['Fuel efficiency'].max())),
                                step=0.5)
        else:
            fe_range = None
    else:
        fe_range = None

# Second row - NEW filters
add_spec_col5, add_spec_col6 = st.columns(2)

with add_spec_col5:
    # NEW: Wheelbase Filter
    if 'Wheel Base mm' in df.columns:
        use_wb_filter = st.checkbox("Filter Wheelbase", value=False,
                                   help="Affects stability and handling")
        if use_wb_filter:
            wb_range = st.slider("Wheelbase (mm)",
                                float(df['Wheel Base mm'].min()),
                                float(df['Wheel Base mm'].max()),
                                (float(df['Wheel Base mm'].min()), float(df['Wheel Base mm'].max())),
                                step=1.0)
        else:
            wb_range = None
    else:
        wb_range = None

# ========= OPTIONAL FILTERS (ENHANCED) ========== #
st.markdown("---")
st.markdown("## 🔧 Optional Filters")
st.warning("⚠️ **Behavioral Economics Warning (H1 & H2)**: Brand filtering may reinforce herd behavior and popularity-driven choices. This framework aims to counter such biases by prioritizing specifications over brand reputation. Use brand filter cautiously and compare with specification-based recommendations.")

opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

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

with opt_col4:
    # NEW: Fuel Type Filter
    use_fuel_type_filter = st.checkbox("Filter by Fuel Type", value=False)
    fuel_type_filter = []
    if use_fuel_type_filter and 'Fuel Type' in df.columns:
        fuel_type_filter = st.multiselect("Fuel Type", sorted(df['Fuel Type'].unique()))

# Second row of optional filters - ALL NEW
opt_col5, opt_col6, opt_col7, opt_col8 = st.columns(4)

with opt_col5:
    # NEW: Front Brake Filter
    use_front_brake_filter = st.checkbox("Filter Front Brake", value=False)
    front_brake_filter = []
    if use_front_brake_filter and 'Front Brake Type' in df.columns:
        front_brake_filter = st.multiselect("Front Brake Type", sorted(df['Front Brake Type'].unique()))

with opt_col6:
    # NEW: Rear Brake Filter
    use_rear_brake_filter = st.checkbox("Filter Rear Brake", value=False)
    rear_brake_filter = []
    if use_rear_brake_filter and 'Rear Brake Type' in df.columns:
        rear_brake_filter = st.multiselect("Rear Brake Type", sorted(df['Rear Brake Type'].unique()))

with opt_col7:
    # NEW: Front Tire Width Filter
    use_front_tire_filter = st.checkbox("Filter Front Tire", value=False)
    front_tire_range = None
    if use_front_tire_filter and 'Front Tyres Size width in mm' in df.columns:
        front_tire_range = st.slider("Front Tire Width (mm)",
                                     float(df['Front Tyres Size width in mm'].min()),
                                     float(df['Front Tyres Size width in mm'].max()),
                                     (float(df['Front Tyres Size width in mm'].min()), 
                                      float(df['Front Tyres Size width in mm'].max())),
                                     step=1.0)

with opt_col8:
    # NEW: Rear Tire Width Filter
    use_rear_tire_filter = st.checkbox("Filter Rear Tire", value=False)
    rear_tire_range = None
    if use_rear_tire_filter and 'Rear Tyres Size width in mm' in df.columns:
        rear_tire_range = st.slider("Rear Tire Width (mm)",
                                    float(df['Rear Tyres Size width in mm'].min()),
                                    float(df['Rear Tyres Size width in mm'].max()),
                                    (float(df['Rear Tyres Size width in mm'].min()), 
                                     float(df['Rear Tyres Size width in mm'].max())),
                                    step=1.0)

# ============= APPLY FILTERS (ENHANCED) ========== #
df_filtered = df.copy()

# Core filters (always applied) - UNCHANGED
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

# Additional spec filters - ENHANCED with new filters
if seat_range:
    df_filtered = df_filtered[(df_filtered['Seat Height mm'] >= seat_range[0]) & 
                              (df_filtered['Seat Height mm'] <= seat_range[1])]
if gc_range:
    df_filtered = df_filtered[(df_filtered['Ground Clearance mm'] >= gc_range[0]) & 
                              (df_filtered['Ground Clearance mm'] <= gc_range[1])]
if fuel_range:
    df_filtered = df_filtered[(df_filtered['Fuel Tank Litre'] >= fuel_range[0]) & 
                              (df_filtered['Fuel Tank Litre'] <= fuel_range[1])]
if fe_range:
    df_filtered = df_filtered[(df_filtered['Fuel efficiency'] >= fe_range[0]) & 
                              (df_filtered['Fuel efficiency'] <= fe_range[1])]
if wb_range:
    df_filtered = df_filtered[(df_filtered['Wheel Base mm'] >= wb_range[0]) & 
                              (df_filtered['Wheel Base mm'] <= wb_range[1])]

# NEW: Tire width filters
if front_tire_range:
    df_filtered = df_filtered[(df_filtered['Front Tyres Size width in mm'] >= front_tire_range[0]) & 
                              (df_filtered['Front Tyres Size width in mm'] <= front_tire_range[1])]
if rear_tire_range:
    df_filtered = df_filtered[(df_filtered['Rear Tyres Size width in mm'] >= rear_tire_range[0]) & 
                              (df_filtered['Rear Tyres Size width in mm'] <= rear_tire_range[1])]

# Optional filters - ENHANCED with new filters
if use_brand_filter and brand_filter:
    df_filtered = df_filtered[df_filtered['Brand'].isin(brand_filter)]
if use_abs_filter and abs_filter:
    df_filtered = df_filtered[df_filtered['ABS'].isin(abs_filter)]
if use_type_filter and type_filter:
    df_filtered = df_filtered[df_filtered['Bike Type'].isin(type_filter)]
if use_fuel_type_filter and fuel_type_filter:
    df_filtered = df_filtered[df_filtered['Fuel Type'].isin(fuel_type_filter)]
if use_front_brake_filter and front_brake_filter:
    df_filtered = df_filtered[df_filtered['Front Brake Type'].isin(front_brake_filter)]
if use_rear_brake_filter and rear_brake_filter:
    df_filtered = df_filtered[df_filtered['Rear Brake Type'].isin(rear_brake_filter)]

st.info(f"🔍 **Bikes after filtering: {len(df_filtered)}** (from {len(df)} total)")

#========== NORMALIZATION (UNCHANGED) ======== #
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
        'Front Tyres Size width in mm', 'Rear Tyres Size width in mm',
        'Front Tyres Ratio in percentage', 'Rear Tyres Ratio in percentage'
    ]
    
    for metric in metrics_to_normalize:
        if metric in df_filtered.columns:
            df_filtered = normalize_column(df_filtered, metric)

#========== SCORING FUNCTIONS (COMPLETELY UNCHANGED) ====== #

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
    """Functional scoring based on bike specifications - NOW INCLUDES FUEL EFFICIENCY & TIRE RATIO"""
    score = 0
    
    # Engine Performance (40 points)
    score += row.get('Engine Displacement_Norm', 0) * 0.13
    score += row.get('Max power PS_Norm', 0) * 0.16
    score += row.get('Max Torque By Nm_Norm', 0) * 0.11
    
    # Fuel Efficiency (5 points)
    score += row.get('Fuel efficiency_Norm', 0) * 0.05
    
    # RPM characteristics (10 points)
    score += row.get('Max power RPM_Norm', 0) * 0.05
    score += row.get('Max Torque RPM_Norm', 0) * 0.05
    
    # Chassis & Handling (30 points)
    score += row.get('Ground Clearance mm_Norm', 0) * 0.10
    score += row.get('Wheel Base mm_Norm', 0) * 0.10
    score += row.get('Fuel Tank Litre_Norm', 0) * 0.10
    
    # Tire Quality (15 points) - Width + Ratio
    front_tire_width = row.get('Front Tyres Size width in mm_Norm', 0) * 0.045
    rear_tire_width = row.get('Rear Tyres Size width in mm_Norm', 0) * 0.045
    # Higher tire ratio = better contact = better grip
    front_tire_ratio = row.get('Front Tyres Ratio in percentage_Norm', 0) * 0.03
    rear_tire_ratio = row.get('Rear Tyres Ratio in percentage_Norm', 0) * 0.03
    score += front_tire_width + rear_tire_width + front_tire_ratio + rear_tire_ratio
    
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
    """Score based on FUNCTIONAL specs matching terrain/purpose (NOT just bike type)
    For Nepali riders: prioritize ground clearance, torque, weight for off-road/mountain
    THESIS FOCUS: Real specification suitability matters more than bike categorization
    """
    if not selected_terrains and not selected_purposes:
        return 50  # Neutral score if no preference
    
    score = 0
    max_possible = 0
    gc = row.get('Ground Clearance mm', 0)
    weight = row.get('Kerb Weight', 0)
    torque = row.get('Max Torque By Nm', 0)
    fuel_eff = row.get('Fuel efficiency', 0)
    
    # SPEC-BASED TERRAIN MATCHING (not type-based)
    if selected_terrains:
        max_possible += 50
        terrain_score = 0
        
        for terrain in selected_terrains:
            if terrain == "Urban/City":
                # City: low seat, light weight, fuel efficient
                seat_fit = 20 if row.get('Seat Height mm', 0) < 800 else 5
                weight_fit = 20 if weight < 140 else 10
                efficiency = 10 if fuel_eff > 30 else 5
                terrain_score += (seat_fit + weight_fit + efficiency) / 3
            
            elif terrain == "Highway/Touring":
                # Highway: comfort, fuel tank, power
                fuel_tank = 15 if row.get('Fuel Tank Litre', 0) >= 12 else 7
                power = 15 if row.get('Max power PS', 0) >= 25 else 8
                seat_comfort = 10 if (750 <= row.get('Seat Height mm', 0) <= 850) else 5
                terrain_score += (fuel_tank + power + seat_comfort) / 3
            
            elif terrain == "Mountain/Off-road":
                # CRITICAL FOR THESIS: Off-road needs GC, torque, light weight
                gc_score = min(30, gc / 5) if gc > 0 else 0  # High GC = high score
                torque_score = min(30, torque * 1.5) if torque > 0 else 0
                weight_score = 20 if weight < 150 else 10  # Light bikes preferred
                terrain_score += (gc_score + torque_score + weight_score) / 3
            
            elif terrain == "Mixed/Versatile":
                # Balanced: moderate on all specs
                gc_score = min(20, gc / 8) if gc > 0 else 0
                torque_score = min(20, torque * 1.2) if torque > 0 else 0
                weight_score = 10 if (130 <= weight <= 170) else 5
                terrain_score += (gc_score + torque_score + weight_score) / 3
        
        score += min(50, (terrain_score / len(selected_terrains)))
    
    # SPEC-BASED PURPOSE MATCHING
    if selected_purposes:
        max_possible += 50
        power = row.get('Max power PS', 0)
        fuel_tank = row.get('Fuel Tank Litre', 0)
        purpose_score = 0
        
        for purpose in selected_purposes:
            if purpose == "Daily Commuting":
                # Commuting: fuel efficient, light, reliable
                efficiency = min(25, fuel_eff * 0.8) if fuel_eff > 0 else 0
                weight_fit = 15 if weight < 140 else 8
                reliability = 10
                purpose_score += (efficiency + weight_fit + reliability) / 3
            
            elif purpose == "Weekend/Sport Riding":
                # Sport: power, torque, responsiveness
                power_score = min(25, power / 2) if power > 0 else 0
                torque_score = min(20, torque * 1.5) if torque > 0 else 0
                weight_score = 5 if weight < 160 else 3
                purpose_score += (power_score + torque_score + weight_score) / 3
            
            elif purpose == "Long Distance Touring":
                # Touring: fuel tank, comfort (seat height), reliability
                tank_score = min(20, fuel_tank * 2) if fuel_tank > 0 else 0
                comfort = 20 if (750 <= row.get('Seat Height mm', 0) <= 850) else 10
                endurance = 10
                purpose_score += (tank_score + comfort + endurance) / 3
            
            elif purpose == "Off-road/Trail":
                # Off-road: ground clearance, torque, light weight (THESIS FOCUS)
                gc_score = min(25, gc / 4) if gc > 0 else 0
                torque_score = min(20, torque * 2) if torque > 0 else 0
                weight_score = 5 if weight < 150 else 2
                purpose_score += (gc_score + torque_score + weight_score) / 3
            
            elif purpose == "Mixed Use":
                # Mixed: balanced across all specs
                avg_score = (min(15, fuel_eff * 0.5) + min(15, power / 3) + 
                            min(15, torque) + min(5, gc / 30)) / 4 if all([fuel_eff, power, torque, gc]) else 0
                purpose_score += avg_score
        
        score += min(50, (purpose_score / len(selected_purposes)))
    
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


# ====CALCULATE SCORES (UNCHANGED) === #
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
    
    # TOTAL SCORE - REWEIGHTED FOR THESIS FOCUS ON FUNCTIONAL SUITABILITY
    # Nepali riders need actual bike specs (GC, torque, weight) to match terrain more than brand fit
    df_filtered['Total Score'] = (
        df_filtered['Ergonomic Score'] * 0.30 +
        df_filtered['Functional Score'] * 0.35 +  # INCREASED: Specs matter most
        df_filtered['Safety Score'] * 0.20 +
        df_filtered['Terrain/Purpose Score'] * 0.10 +  # REDUCED: Was overweighting bike type classification
        df_filtered['Value Score'] * 0.05  # REDUCED: Spec match > price match for thesis focus
    )
    
    # ===== PURPOSE-SPECIFIC OVERALL SCORES ===== #
    # Calculate overall scores for different riding purposes
    def calculate_purpose_score(row, purpose_type):
        """Calculate overall score optimized for specific purpose"""
        if purpose_type == "urban_commute":
            # Urban: Ergonomic + Fuel Efficiency + Safety + Value
            return (row['Ergonomic Score'] * 0.35 + row['Functional Score'] * 0.20 + 
                   row['Safety Score'] * 0.25 + row['Value Score'] * 0.20)
        elif purpose_type == "mountain_offroad":
            # Mountain: Functional (GC, Torque) + Terrain + Ergonomic + Safety
            return (row['Terrain/Purpose Score'] * 0.40 + row['Functional Score'] * 0.30 + 
                   row['Ergonomic Score'] * 0.20 + row['Safety Score'] * 0.10)
        elif purpose_type == "touring":
            # Touring: Functional (fuel tank, power) + Ergonomic (comfort) + Value
            return (row['Functional Score'] * 0.35 + row['Ergonomic Score'] * 0.30 + 
                   row['Value Score'] * 0.20 + row['Safety Score'] * 0.15)
        elif purpose_type == "sport":
            # Sport: Functional (power, torque) + Terrain + Safety + Ergonomic
            return (row['Functional Score'] * 0.40 + row['Safety Score'] * 0.25 + 
                   row['Terrain/Purpose Score'] * 0.20 + row['Ergonomic Score'] * 0.15)
        else:
            return row['Total Score']
    
    df_filtered['Urban/Commute Score'] = df_filtered.apply(lambda row: calculate_purpose_score(row, "urban_commute"), axis=1)
    df_filtered['Mountain/Off-road Score'] = df_filtered.apply(lambda row: calculate_purpose_score(row, "mountain_offroad"), axis=1)
    df_filtered['Touring Score'] = df_filtered.apply(lambda row: calculate_purpose_score(row, "touring"), axis=1)
    df_filtered['Sport Score'] = df_filtered.apply(lambda row: calculate_purpose_score(row, "sport"), axis=1)
    
    # Sort by total score
    df_filtered = df_filtered.sort_values('Total Score', ascending=False)

# ==================== RESULTS DISPLAY (PURPOSE-SPECIFIC TABS) ==================== #
st.markdown("---")
st.markdown("## 🏆 All Recommendations Ranked by Score")
st.info("📊 **Showing ALL bikes** that match your filters, sorted by score (highest to lowest). Choose your priority below:")

# Show dataset coverage info
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
with info_col1:
    st.metric("Total Available Bikes", len(df))
with info_col2:
    st.metric("Currently Showing", len(df_filtered))
with info_col3:
    st.metric("Filtered Out", len(df) - len(df_filtered))
with info_col4:
    coverage = (len(df_filtered) / len(df) * 100) if len(df) > 0 else 0
    st.metric("Coverage %", f"{coverage:.1f}%")

if df_filtered.empty:
    st.error("❌ No bikes match your filters. Try widening your criteria.")
else:
    # ===== PURPOSE-SPECIFIC RECOMMENDATIONS ===== #
    purpose_tabs = st.tabs(["🏆 Overall", "🏙️ Urban/Commute", "⛰️ Mountain/Off-road", "🏍️ Touring", "🏁 Sport"])
    
    with purpose_tabs[0]:
        st.markdown("### All Recommendations (Balanced Overall Score)")
        top_15 = df_filtered.sort_values('Total Score', ascending=False)  # Show ALL bikes
        score_cols = ['Total Score', 'Ergonomic Score', 'Functional Score', 
                      'Safety Score', 'Terrain/Purpose Score', 'Value Score']
    
    with purpose_tabs[1]:
        st.markdown("### All Recommendations (Urban/Commute Optimized)")
        st.caption("🏙️ Score: Seat height (35%) + Engine efficiency (20%) + Safety (25%) + Value (20%)")
        top_15 = df_filtered.sort_values('Urban/Commute Score', ascending=False)  # Show ALL bikes
        score_cols = ['Urban/Commute Score', 'Ergonomic Score', 'Functional Score', 
                      'Safety Score', 'Value Score']
        st.info("🏙️ Perfect for Kathmandu Valley: Low seat for traffic, fuel efficient, responsive brakes")
    
    with purpose_tabs[2]:
        st.markdown("### All Recommendations (Mountain/Off-road Optimized)")
        st.caption("⛰️ Score: Ground clearance & torque (40%) + Terrain (20%) + Ergonomic (20%) + Safety (10%)")
        top_15 = df_filtered.sort_values('Mountain/Off-road Score', ascending=False)  # Show ALL bikes
        score_cols = ['Mountain/Off-road Score', 'Terrain/Purpose Score', 'Functional Score', 
                      'Ergonomic Score', 'Safety Score']
        st.info("⛰️ For mountain passes & monsoon: High GC for rough roads, torque for climbs, light weight")
    
    with purpose_tabs[3]:
        st.markdown("### All Recommendations (Touring Optimized)")
        st.caption("🏍️ Score: Fuel tank & power (35%) + Comfort (30%) + Value (20%) + Safety (15%)")
        top_15 = df_filtered.sort_values('Touring Score', ascending=False)  # Show ALL bikes
        score_cols = ['Touring Score', 'Functional Score', 'Ergonomic Score', 
                      'Value Score', 'Safety Score']
        st.info("🏍️ For long distance: Large fuel tank, comfortable ergonomics, good power, reasonable price")
    
    with purpose_tabs[4]:
        st.markdown("### All Recommendations (Sport Riding Optimized)")
        st.caption("🏁 Score: Power & torque (40%) + Safety/brakes (25%) + Handling (20%) + Ergonomic (15%)")
        top_15 = df_filtered.sort_values('Sport Score', ascending=False)  # Show ALL bikes
        score_cols = ['Sport Score', 'Functional Score', 'Safety Score', 
                      'Terrain/Purpose Score', 'Ergonomic Score']
        st.info("🏁 For sport riding: High power/torque, responsive handling, strong brakes, lightweight")
    
    # Display ALL columns - scores first, then all bike specs
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
            try:
                if 'Price' in col:
                    format_dict[col] = '{:,.0f}'
                elif 'Score' in col:
                    format_dict[col] = '{:.1f}'
                elif top_15[col].dtype in ['float64', 'float32']:
                    format_dict[col] = '{:.1f}'
                elif top_15[col].dtype in ['int64', 'int32']:
                    format_dict[col] = '{:.0f}'
            except:
                pass
    
    # Format and display dataframe
    try:
        styled_df = top_15[display_cols].style.format(format_dict).background_gradient(
            subset=[col for col in ['Total Score'] if col in display_cols], 
            cmap='Greens'
        )
        st.dataframe(styled_df, width='stretch')
    except:
        st.dataframe(top_15[display_cols], width='stretch')
    
    # Download option
    csv = top_15[display_cols].to_csv(index=False)
    st.download_button(
        label=f"📥 Download All {len(top_15)} Recommendations (CSV)",
        data=csv,
        file_name="bike_recommendations.csv",
        mime="text/csv"
    )
    
    # ===== VISUALIZATIONS (UNCHANGED) === #
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
    
    # ========== DETAILED COMPARISON (UNCHANGED) ====== #
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
            
            try:
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
            except Exception as e:
                st.warning(f"Could not create radar chart: {str(e)}")
        
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
        
        st.dataframe(comparison_df[comparison_cols], width='stretch')
        
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
    
    # == INSIGHTS & RECOMMENDATIONS  == #
    st.markdown("---")
    st.markdown("## 💡 Personalized Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("### Best Ergonomic Fit")
        best_ergo = df_filtered.nlargest(3, 'Ergonomic Score')[['Bike Names', 'Ergonomic Score', 'Seat Height mm', 'Kerb Weight']]
        st.dataframe(best_ergo, width='stretch')
        
        st.markdown("### Best Value for Money")
        best_value = df_filtered.nlargest(3, 'Value Score')[['Bike Names', 'Value Score', 'Price (Rs)']]
        st.dataframe(best_value, width='stretch')
    
    with insight_col2:
        st.markdown("### Best Performance")
        best_func = df_filtered.nlargest(3, 'Functional Score')[['Bike Names', 'Functional Score', 'Max power PS', 'Max Torque By Nm']]
        st.dataframe(best_func, width='stretch')
        
        st.markdown("### Safest Options")
        best_safety = df_filtered.nlargest(3, 'Safety Score')[['Bike Names', 'Safety Score', 'ABS']]
        st.dataframe(best_safety, width='stretch')
    
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

# ===== VIEW ALL BIKES ===== #
st.markdown("---")
st.markdown("## 📋 View All Bikes in Dataset")

with st.expander("📊 Show All Bikes with Full Specifications", expanded=False):
    st.info(f"✅ Total bikes in dataset: {len(df)} | Bikes matching current filters: {len(df_filtered)}")
    
    # Allow user to view either all bikes or filtered bikes
    view_choice = st.radio("What would you like to view?", 
                          ["All Bikes in Dataset", "Currently Filtered Bikes"],
                          horizontal=True)
    
    if view_choice == "All Bikes in Dataset":
        view_df = df.sort_values('Bike Names')
    else:
        view_df = df_filtered.sort_values('Bike Names')
    
    # Show full bike details - ALL 23 COLUMNS
    all_cols = ['Bike Names', 'Brand', 'Bike Type', 'Price (Rs)', 
                'Engine Displacement', 'Max power PS', 'Max power RPM', 
                'Max Torque By Nm', 'Max Torque RPM',
                'Kerb Weight', 'Seat Height mm', 'Ground Clearance mm',
                'Fuel Tank Litre', 'Wheel Base mm',
                'Front Tyres Size width in mm', 'Front Tyres Ratio in percentage',
                'Rear Tyres Size width in mm', 'Rear Tyres Ratio in percentage',
                'Front Brake Type', 'Rear Brake Type', 'ABS',
                'Fuel efficiency', 'Fuel efficiency type']
    
    all_cols = [col for col in all_cols if col in view_df.columns]
    
    st.dataframe(view_df[all_cols], width='stretch')
    
    # Download all bikes
    csv_all = view_df[all_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download All Bikes Data (CSV)",
        data=csv_all,
        file_name="all_bikes_dataset.csv",
        mime="text/csv"
    )

# ==== FOOTER ==== #
st.markdown("---")
st.markdown("""
### 📚 About This Framework

**Thesis Research**: This decision support framework is part of academic research on motorcycle selection behavior among urban youth (18-25) in Nepal.

**Theoretical Foundation**:
- **Multi-Criteria Decision Analysis (MCDA)**: Structures complex trade-offs across multiple criteria
- **Behavioral Economics**: Addresses herd behavior, status-driven consumption, and cognitive biases
- **Decision Theory**: Supports bounded rationality without assuming perfect information processing

**Key Insights**:
1. **Herd Behavior**: Young riders often follow peer choices rather than evaluating specifications
2. **Status Signaling**: Motorcycles serve as social identity markers, not just transportation
3. **Information Overload**: 20+ specifications across 100+ models create cognitive burden
4. **Bounded Rationality**: Structured frameworks extend decision-making capacity

**Ethical Considerations**:
- Framework informs rather than dictates choices
- Transparency in criteria weights and scoring logic
- Acknowledges cultural values (status, peer influence) while highlighting functional trade-offs
- Designed to empower autonomous decision-making, not replace it

**Limitations**:
- Conceptual framework not empirically validated through user studies
- Thresholds and weights based on literature review, not field testing
- Missing ergonomic data (handlebar reach, knee angles, riding position)
- Does not account for rider experience level or learning curve

**Research Status**: Literature review complete. Implementation phase (this dashboard) demonstrates conceptual framework. Next phase would involve empirical validation with urban Nepali youth.

---

**Version**: 3.0 (Thesis-Aligned)  
**Last Updated**: January 2026  
**Target Users**: Urban Youth (18-25) in Nepal  
**Price Range**: 4 Lakhs+ (₹400,000+)
""")



