import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
# import joblib # Removed as it is unused in the current code

# --- Configuration and Constants ---
# We are using the monthly, sector-specific data (IPCC 2006)
FILE_PATH = "IEA_EDGAR_CO2_m_1970_2023.csv"
SKIP_ROWS = 9 
MONTH_COLUMNS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
TARGET_COLUMN = 'Emissions_Gg' 
# MODELING_SAMPLE_SIZE removed - using full dataset for better accuracy

# --- Data Loading and Preprocessing (Cached for Performance) ---

@st.cache_data(show_spinner="Loading and transforming the massive dataset...")
def load_and_preprocess_data(file_path):
    """Loads, cleans, and transforms the monthly CO2 emissions data, handling common encoding issues."""
    
    # Expanded list of common encodings to try. Added utf-16le and utf-32.
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-16le', 'utf-32']
    
    df = None
    successful_encoding = None
    
    # Attempt to read the file with different encodings
    for encoding in encodings_to_try:
        try:
            # 1. Load Data with specified encoding
            df = pd.read_csv(file_path, skiprows=SKIP_ROWS, encoding=encoding)
            successful_encoding = encoding
            st.success(f"Data loaded successfully using {encoding} encoding.")
            break # Exit loop if successful
        except FileNotFoundError:
            # File not found error is critical and should stop the process immediately
            st.error(f"Error: Data file not found at {file_path}. Please ensure the file is correctly named and accessible.")
            return None
        except Exception as e:
            # If decoding fails, try the next encoding
            st.warning(f"Decoding failed with {encoding}. Trying next encoding...")
            continue
            
    if df is None:
        st.error(f"Failed to load data. All attempted encodings failed: {', '.join(encodings_to_try)}.")
        st.warning("Please verify the file integrity or try converting the CSV file encoding manually (e.g., using Notepad++ or Excel) to UTF-8 before uploading.")
        return None

    try:
        # 2. Select Core Columns 
        # Note: We expect 'ipcc_code_2006_for_standard_report_name' to be in the headers
        core_cols = ['Name', 'ipcc_code_2006_for_standard_report_name', 'Year'] + MONTH_COLUMNS
        df = df[core_cols]

        # 3. Data Transformation (Melt monthly columns to long format)
        df_long = pd.melt(
            df, 
            id_vars=['Name', 'ipcc_code_2006_for_standard_report_name', 'Year'],
            value_vars=MONTH_COLUMNS,
            var_name='Month_Name',
            value_name=TARGET_COLUMN
        )

        # 4. Feature Engineering
        month_map = {name: i + 1 for i, name in enumerate(MONTH_COLUMNS)}
        df_long['Month_Num'] = df_long['Month_Name'].map(month_map)
        
        # 5. Clean Data
        df_long = df_long.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET_COLUMN])
        
        return df_long
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing after successful load: {e}")
        # Display columns for debugging if processing fails
        st.write("Columns found in the successfully loaded raw data:", df.columns.tolist())
        return None


# --- Model Training (Cached for Performance) ---

@st.cache_resource(show_spinner="Training the Random Forest Model on full dataset (This will take longer)...")
def train_random_forest_model(df_full):
    """Trains the Random Forest Regressor on the full dataset for max accuracy."""
    
    st.info(f"Using the **full dataset** of {len(df_full):,} records for maximum model accuracy.")
    df_modeling = df_full.copy()
    
    # Define minimum year for Time_Index calculation
    MIN_YEAR = df_modeling['Year'].min()
        
    # Initialize LabelEncoders
    le_country = LabelEncoder()
    le_sector = LabelEncoder()
    
    # Check for required columns before encoding
    required_cols = ['Name', 'ipcc_code_2006_for_standard_report_name']
    if not all(col in df_modeling.columns for col in required_cols):
        st.error("Missing required column for encoding. Data processing failed.")
        return None, None, None, None, None, None, None

    # --- Feature Engineering Enhancement ---
    # Add a time index feature
    df_modeling['Time_Index'] = df_modeling['Year'] - MIN_YEAR
    
    # Apply encoding
    df_modeling['Country_Encoded'] = le_country.fit_transform(df_modeling['Name'])
    df_modeling['Sector_Encoded'] = le_sector.fit_transform(df_modeling['ipcc_code_2006_for_standard_report_name'])

    # Define features and target
    # Added 'Time_Index' to the feature list
    features = ['Year', 'Time_Index', 'Month_Num', 'Country_Encoded', 'Sector_Encoded']
    X = df_modeling[features]
    y = df_modeling[TARGET_COLUMN]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    # Increased n_estimators and max_depth for better complexity and non-static predictions
    rf_model = RandomForestRegressor(
        n_estimators=100, # Increased from 50
        random_state=42, 
        n_jobs=-1, 
        max_depth=25 # Increased from 15 to force deeper splits and capture subtle time trends
    )
    # 
    rf_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rf_model, le_country, le_sector, r2, rmse, features, MIN_YEAR

# --- Visualization Functions (No changes needed here) ---

def plot_annual_trend(df):
    """Generates the Annual CO2 Emissions Trend plot."""
    annual_emissions = df.groupby('Year')[TARGET_COLUMN].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=annual_emissions, x='Year', y=TARGET_COLUMN, marker='o', color='darkblue', ax=ax)
    ax.set_title('Global Annual CO2 Emissions Trend (1970-2023)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total CO2 Emissions (Gg)')
    st.pyplot(fig)

def plot_monthly_seasonality(df):
    """Generates the Monthly CO2 Emissions Seasonality plot."""
    monthly_emissions_avg = df.groupby('Month_Name')[TARGET_COLUMN].mean().reset_index()
    month_map = {name: i + 1 for i, name in enumerate(MONTH_COLUMNS)}
    monthly_emissions_avg['Month_Num'] = monthly_emissions_avg['Month_Name'].map(month_map)
    monthly_emissions_avg = monthly_emissions_avg.sort_values('Month_Num')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=monthly_emissions_avg, x='Month_Name', y=TARGET_COLUMN, palette='viridis', ax=ax)
    ax.set_title('Global Average Monthly CO2 Emissions (Seasonality)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average CO2 Emissions (Gg)')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def plot_sector_trends(df):
    """Generates the Top 10 CO2 Emitting Sectors plot."""
    sector_emissions = df.groupby('ipcc_code_2006_for_standard_report_name')[TARGET_COLUMN].sum().sort_values(ascending=False).head(10).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=sector_emissions, y='ipcc_code_2006_for_standard_report_name', x=TARGET_COLUMN, palette='magma', ax=ax)
    ax.set_title('Top 10 CO2 Emitting Sectors (Total Emissions)')
    ax.set_xlabel('Total CO2 Emissions (Gg)')
    ax.set_ylabel('IPCC Sector Name')
    st.pyplot(fig)

def plot_feature_importance(model, features):
    """Plots the feature importance from the trained Random Forest model."""
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=forest_importances.values, y=forest_importances.index, palette='Reds_d', ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Feature Importance Score (Gini Importance)")
    st.pyplot(fig)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Advanced CO2 Emissions Analytics")

# Load and preprocess data once
df = load_and_preprocess_data(FILE_PATH)

if df is not None:
    
    st.title("🌍 Advanced CO2 Emissions Analytics & Prediction")
    st.markdown("A data science project using the IEA/EDGAR CO2 dataset to analyze global emissions trends and predict future levels using a Random Forest model.")
    
    # --- Sidebar for Navigation/Info ---
    st.sidebar.header("Project Details")
    st.sidebar.markdown(f"**Data Source:** IEA/EDGAR CO2 (Monthly, 1970-2023)")
    st.sidebar.markdown(f"**Total Records:** {len(df):,}")
    st.sidebar.markdown(f"**Modeling Algorithm:** Random Forest Regression (Trained on Full Data)")
    
    # --- Section 1: Data Analysis and Trends ---
    st.header("1. Emissions Trend Analysis")
    st.markdown("Visualizing the required monthly, annual, and sector-wise trends.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annual Trend")
        plot_annual_trend(df)
    
    with col2:
        st.subheader("Monthly Seasonality")
        plot_monthly_seasonality(df)
        st.info("The monthly plot reveals global seasonal fluctuations in CO2 output.")

    st.subheader("Top Emitting Sectors")
    plot_sector_trends(df)
    st.markdown("This chart highlights the ten sectors that are the largest cumulative contributors to global CO2 emissions.")

    st.markdown("---")

    # --- Section 2: Random Forest Model Training & Evaluation ---
    st.header("2. Random Forest Model")

    # Train model components and cache them
    model_results = train_random_forest_model(df)
    
    if model_results is not None and model_results[0] is not None:
        rf_model, le_country, le_sector, r2, rmse, features, MIN_YEAR = model_results
        
        st.subheader("Model Performance")
        colA, colB = st.columns(2)
        colA.metric(label="R-squared (R²)", value=f"{r2:.4f}", help="Measures how well the independent variables explain the variance in the dependent variable (0.0 to 1.0).")
        colB.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:,.2f} Gg", help="The standard deviation of the residuals (prediction errors).")
        
        st.subheader("Feature Importance")
        st.markdown("The Random Forest model determines the relative importance of each input feature in predicting the emissions value.")
        plot_feature_importance(rf_model, features)
        st.success("Training on the full dataset and using the enhanced Time Index should lead to more accurate, year-dependent predictions.")

        st.markdown("---")
        
        # --- Section 3: Future Prediction Tool ---
        st.header("3. Future Emissions Prediction")
        st.markdown("Use the trained Random Forest Regressor to predict CO2 emissions for a specific future scenario.")

        # Get available options from the encoders
        available_countries = sorted(list(le_country.classes_))
        available_sectors = sorted(list(le_sector.classes_))
        
        # Add 'All' options
        country_options = ['All Countries'] + available_countries
        sector_options = ['All Sectors'] + available_sectors
        month_options = ['All Months (Annual Average)'] + MONTH_COLUMNS
        
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            # Default to All Countries
            selected_country = st.selectbox("Select Country:", country_options, index=0)
        
        with col_pred2:
            # Default to All Sectors
            selected_sector = st.selectbox("Select Sector:", sector_options, index=0)
            
        with col_pred3:
            # Set slider max to 2050
            current_year = pd.Timestamp.now().year
            selected_year = st.slider("Select Year:", min_value=current_year + 1, max_value=2050, value=2030)
            # Default to All Months (Index 0)
            selected_month = st.selectbox("Select Month:", options=month_options, index=0) 
        
        if st.button("Generate Prediction", type="primary"):
            
            # --- Handle 'All' Selections and Prediction ---
            
            # 1. Determine which values to predict over
            
            # Country Encoding: Use the mean of all encoded values if 'All Countries' is selected
            if selected_country == 'All Countries':
                # Calculate the average of all country encodings for a 'global average country' impact
                country_encodings = le_country.transform(available_countries)
                country_encoded_value = country_encodings.mean()
            else:
                country_encoded_value = le_country.transform([selected_country])[0]
                
            # Sector Encoding: Use the mean of all encoded values if 'All Sectors' is selected
            if selected_sector == 'All Sectors':
                # Calculate the average of all sector encodings for a 'global average sector' impact
                sector_encodings = le_sector.transform(available_sectors)
                sector_encoded_value = sector_encodings.mean()
            else:
                sector_encoded_value = le_sector.transform([selected_sector])[0]
            
            # Month Prediction: If 'All Months' is selected, predict for all 12 months and take the average
            if selected_month == 'All Months (Annual Average)':
                month_nums = np.arange(1, 13) # Months 1 through 12
                prediction_label = f"Annual Average (Gg CO₂)"
            else:
                month_nums = [MONTH_COLUMNS.index(selected_month) + 1]
                prediction_label = f"Monthly (Gg CO₂)"
            
            # --- Feature preparation for Prediction ---
            # Use the Time Index feature relative to the MIN_YEAR (1970)
            time_index_value = selected_year - MIN_YEAR
            
            # 2. Create input DataFrame for prediction
            new_data = pd.DataFrame({
                'Year': selected_year,
                'Time_Index': time_index_value, 
                'Month_Num': month_nums,
                'Country_Encoded': country_encoded_value,
                'Sector_Encoded': sector_encoded_value
            })
            
            # 3. Predict and Aggregate
            predictions = rf_model.predict(new_data)
            
            if selected_month == 'All Months (Annual Average)':
                # Calculate the average of the 12 monthly predictions
                final_prediction = predictions.mean()
                prediction_text = f"Predicted **Average Monthly** Emissions in **{selected_year}**"
                prediction_unit = f"Gg CO₂ / month"
            else:
                final_prediction = predictions[0]
                prediction_text = f"Predicted Emissions in **{selected_month} {selected_year}**"
                prediction_unit = f"Gg CO₂"

            # 4. Display Result
            st.success("--- Prediction Result ---")
            st.markdown(f"**Scenario:** Country: **{selected_country}**, Sector: **{selected_sector}**")
            st.markdown(prediction_text)
            st.markdown(f"## **{final_prediction:,.2f} {prediction_unit}**")
            st.caption("Gg = Gigagrams, equivalent to 1,000 tonnes.")

            # Note about the "All" calculation 
            if selected_country == 'All Countries' or selected_sector == 'All Sectors':
                st.info(f"**Note on Aggregation:** When 'All Countries' or 'All Sectors' is selected, the prediction represents the result using the *average encoded value* for that feature. This models the emissions of an 'average entity' rather than summing all entities, which is generally more robust for Random Forest prediction using Label Encoding.")

            
    else:
        st.error("Model training failed. This can happen if the sampled data is too small or if required columns are missing.")

else:
    st.error("\nApplication halted due to data loading failure. Please check the file path, name, and ensure the data file is accessible.")