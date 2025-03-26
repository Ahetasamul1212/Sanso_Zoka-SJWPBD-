import streamlit as st
import pandas as pd
import numpy as np
import time
import uuid
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="DO Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0083B8;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0083B8;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-result {
        background-color: #e6f7ff;
        border-left: 5px solid #0083B8;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #0083B8;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #006491;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .nav-button {
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e6f7ff;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .warning {
        color: #ff4b4b;
        font-weight: bold;
    }
    .success {
        color: #00c853;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# File to store predictions
PREDICTIONS_FILE = "predictions.csv"

# Load dataset
@st.cache_data
def load_dataset():
    file_path = "EDA_Model/Model_test_data/DOE.csv"
    try:
        df = pd.read_csv(file_path, encoding="latin1")
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        return None

df = load_dataset()
if df is None:
    st.stop()

# Data Preprocessing
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    if "EC(µS/cm)" in df_processed.columns:
        df_processed["EC(µS/cm)"] = pd.to_numeric(df_processed["EC(µS/cm)"], errors="coerce")

    if "Date" in df_processed.columns:
        df_processed["Date"] = pd.to_datetime(df_processed["Date"], errors="coerce")

    num_cols = df_processed.select_dtypes(include=["number"]).columns
    df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())

    columns_to_drop = ["Sample Location", "Lab code", "Date"]
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors="ignore")
    
    return df_processed

df_processed = preprocess_data(df)

# Define features and target variable
X = df_processed.drop(columns=["DO(mg/l)"])
y = df_processed["DO(mg/l)"]

# Train model
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Load existing predictions
def load_predictions():
    if os.path.exists(PREDICTIONS_FILE):
        return pd.read_csv(PREDICTIONS_FILE)
    return pd.DataFrame(columns=["ID", "Timestamp", *X.columns, "Predicted DO (mg/l)"])

# Save predictions
def save_prediction(prediction_entry):
    df_pred = load_predictions()
    df_pred = pd.concat([df_pred, pd.DataFrame([prediction_entry])], ignore_index=True)
    df_pred.to_csv(PREDICTIONS_FILE, index=False)

# Feature importance function
@st.cache_data
def get_feature_importance():
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    return feature_importance

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar with logo and navigation
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    
    # Logo/Header
    st.image("https://v0.placeholder.svg?height=100&width=300&text=DO+Prediction+System", use_column_width=True)
    st.markdown("---")
    
    st.markdown("### Navigation")
    
    # Navigation buttons with icons
    if st.button("🏠 Home", key="home_btn", use_container_width=True):
        st.session_state.page = "Home"
    
    if st.button("🔮 Predict", key="predict_btn", use_container_width=True):
        st.session_state.page = "Predict"
        
    if st.button("🗃️ Database", key="database_btn", use_container_width=True):
        st.session_state.page = "Database"
        
    if st.button("📊 Analytics", key="analytics_btn", use_container_width=True):
        st.session_state.page = "Analytics"
        
    if st.button("ℹ️ About", key="about_btn", use_container_width=True):
        st.session_state.page = "About"
    
    st.markdown("---")
    st.markdown("### System Info")
    st.info(f"Model: Random Forest Regressor\nFeatures: {len(X.columns)}\nSamples: {len(df_processed)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Home Page
if st.session_state.page == "Home":
    st.markdown("<h1 class='main-header'>Dissolved Oxygen (DO) Prediction System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Welcome to the DO Prediction System")
        st.markdown("""
        This advanced system uses machine learning to predict Dissolved Oxygen levels in water bodies based on various water quality parameters.
        
        **Key Features:**
        - Accurate prediction using Random Forest algorithm
        - Historical prediction database
        - Data visualization and analytics
        - User-friendly interface
        
        Use the navigation panel on the left to explore different sections of the application.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Quick Start Guide")
        st.markdown("""
        1. Go to the **Predict** page to input water quality parameters
        2. View your prediction history in the **Database** section
        3. Explore data insights in the **Analytics** section
        4. Learn more about the project in the **About** section
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### What is Dissolved Oxygen?")
        st.markdown("""
        Dissolved Oxygen (DO) is the amount of oxygen present in water. It's a crucial indicator of water quality and aquatic life health.
        
        **Importance:**
        - Essential for aquatic organisms
        - Indicator of pollution levels
        - Critical for ecosystem balance
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Recent Activity")
        
        df_pred = load_predictions()
        if not df_pred.empty:
            st.markdown(f"**Total predictions:** {len(df_pred)}")
            st.markdown(f"**Latest prediction:** {df_pred['Timestamp'].iloc[-1] if 'Timestamp' in df_pred.columns else 'N/A'}")
        else:
            st.markdown("No predictions made yet.")
        st.markdown("</div>", unsafe_allow_html=True)

# Prediction Page
elif st.session_state.page == "Predict":
    st.markdown("<h1 class='main-header'>Dissolved Oxygen Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Enter Water Quality Parameters")
    st.markdown("Adjust the sliders below to input the water quality parameters for prediction.")
    
    # Create a more organized layout for inputs
    col1, col2 = st.columns(2)
    user_input = {}
    
    # Split features into two columns for better organization
    features_list = list(X.columns)
    half = len(features_list) // 2
    
    for i, col in enumerate(features_list[:half]):
        with col1:
            min_val = float(df_processed[col].min())
            max_val = float(df_processed[col].max())
            med_val = float(df_processed[col].median())
            user_input[col] = st.slider(
                f"{col}", 
                min_val, 
                max_val, 
                med_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}, Median: {med_val:.2f}"
            )
    
    for i, col in enumerate(features_list[half:]):
        with col2:
            min_val = float(df_processed[col].min())
            max_val = float(df_processed[col].max())
            med_val = float(df_processed[col].median())
            user_input[col] = st.slider(
                f"{col}", 
                min_val, 
                max_val, 
                med_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f}, Median: {med_val:.2f}"
            )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("Predict DO Level", use_container_width=True)
    
    if predict_button:
        with st.spinner("Processing prediction..."):
            # Progress bar animation
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Make prediction
            user_df = pd.DataFrame([user_input])
            prediction = model.predict(user_df)
            
            # Generate unique ID and timestamp
            unique_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Prepare prediction entry
            prediction_entry = {
                "ID": unique_id, 
                "Timestamp": timestamp,
                **user_input, 
                "Predicted DO (mg/l)": round(prediction[0], 2)
            }
            
            # Save prediction
            save_prediction(prediction_entry)
            
            # Clear progress bar
            progress_bar.empty()
        
        # Display prediction result
        st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Predicted DO Level: <span class='success'>{prediction[0]:.2f} mg/l</span></h2>", unsafe_allow_html=True)
        
        # Interpret the result
        if prediction[0] < 2:
            interpretation = "Very Low - Critical condition for aquatic life"
            status_color = "red"
        elif prediction[0] < 4:
            interpretation = "Low - Stressful for most aquatic organisms"
            status_color = "orange"
        elif prediction[0] < 7:
            interpretation = "Moderate - Acceptable for most species"
            status_color = "blue"
        else:
            interpretation = "High - Excellent conditions for aquatic life"
            status_color = "green"
        
        st.markdown(f"<p style='font-size: 1.2rem;'>Interpretation: <span style='color:{status_color};'>{interpretation}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p>Prediction ID: {unique_id}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>Timestamp: {timestamp}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show feature importance for this prediction
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Feature Contribution to Prediction")
        
        # Get feature importance
        feature_importance = get_feature_importance()
        
        # Create a horizontal bar chart
        fig = px.bar(
            feature_importance.head(10), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 10 Features by Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Database Page
elif st.session_state.page == "Database":
    st.markdown("<h1 class='main-header'>Prediction Database</h1>", unsafe_allow_html=True)
    
    df_predictions = load_predictions()
    
    if not df_predictions.empty:
        # Add search and filter options
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Search and Filter")
        
        # Date filter if timestamp column exists
        if "Timestamp" in df_predictions.columns:
            df_predictions["Timestamp"] = pd.to_datetime(df_predictions["Timestamp"])
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", df_predictions["Timestamp"].min().date())
            with date_col2:
                end_date = st.date_input("End Date", df_predictions["Timestamp"].max().date())
            
            # Filter by date
            mask = (df_predictions["Timestamp"].dt.date >= start_date) & (df_predictions["Timestamp"].dt.date <= end_date)
            df_filtered = df_predictions.loc[mask]
        else:
            df_filtered = df_predictions
        
        # Search by ID
        search_id = st.text_input("Search by Prediction ID")
        if search_id:
            df_filtered = df_filtered[df_filtered["ID"].str.contains(search_id, case=False)]
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display database
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Prediction Records")
        st.markdown(f"Showing {len(df_filtered)} of {len(df_predictions)} records")
        
        # Display the dataframe with styling
        st.dataframe(
            df_filtered.style.background_gradient(subset=["Predicted DO (mg/l)"], cmap="Blues"),
            use_container_width=True,
            height=400
        )
        
        # Export options
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("Export to CSV", use_container_width=True):
                csv = df_filtered.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="do_predictions.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with export_col2:
            if st.button("Clear Database", use_container_width=True):
                if os.path.exists(PREDICTIONS_FILE):
                    os.remove(PREDICTIONS_FILE)
                    st.success("Database cleared successfully!")
                    st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.warning("No predictions have been made yet. Go to the Predict page to make predictions.")
        st.markdown("</div>", unsafe_allow_html=True)

# Analytics Page (New)
elif st.session_state.page == "Analytics":
    st.markdown("<h1 class='main-header'>Data Analytics</h1>", unsafe_allow_html=True)
    
    # Feature importance analysis
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Feature Importance Analysis")
    
    feature_importance = get_feature_importance()
    
    # Create a horizontal bar chart
    fig = px.bar(
        feature_importance, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Features Ranked by Importance',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    This chart shows the relative importance of each feature in predicting the Dissolved Oxygen level.
    Features with higher importance have a greater impact on the prediction results.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Correlation analysis
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Correlation Analysis")
    
    # Calculate correlation matrix
    corr_matrix = df_processed.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    The correlation matrix shows relationships between different water quality parameters.
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate little to no correlation
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction history analysis
    df_predictions = load_predictions()
    if not df_predictions.empty and "Timestamp" in df_predictions.columns:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Prediction History Analysis")
        
        # Convert timestamp to datetime if it's not already
        df_predictions["Timestamp"] = pd.to_datetime(df_predictions["Timestamp"])
        
        # Add date column for grouping
        df_predictions["Date"] = df_predictions["Timestamp"].dt.date
        
        # Group by date and calculate average DO prediction
        daily_avg = df_predictions.groupby("Date")["Predicted DO (mg/l)"].mean().reset_index()
        
        # Create line chart
        fig = px.line(
            daily_avg,
            x="Date",
            y="Predicted DO (mg/l)",
            title="Average Daily DO Predictions",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("This chart shows the trend of average daily DO predictions over time.")
        st.markdown("</div>", unsafe_allow_html=True)

# About Page
elif st.session_state.page == "About":
    st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Project Overview")
        st.markdown("""
        The **Dissolved Oxygen (DO) Prediction System** is an advanced machine learning application designed to predict dissolved oxygen levels in water bodies based on various water quality parameters.
        
        This system utilizes a Random Forest Regression model trained on historical water quality data to make accurate predictions of DO levels, which are crucial indicators of water quality and aquatic ecosystem health.
        
        The application provides a user-friendly interface for making predictions, storing results, and analyzing data patterns to support water quality monitoring and management efforts.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Technical Details")
        st.markdown("""
        **Technologies Used:**
        - **Python**: Core programming language
        - **Streamlit**: Web application framework
        - **Pandas & NumPy**: Data processing and analysis
        - **Scikit-learn**: Machine learning implementation (Random Forest Regressor)
        - **Plotly & Matplotlib**: Data visualization
        - **Seaborn**: Statistical data visualization
        
        **Model Information:**
        - **Algorithm**: Random Forest Regressor
        - **Features**: Multiple water quality parameters
        - **Target Variable**: Dissolved Oxygen (DO) in mg/l
        - **Evaluation Metric**: Mean Squared Error (MSE)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Development Team")
        st.markdown("""
        **Developed by:**
        - **Mohammed Ahtesamul Rasul**
        - **Ilham Sajid**
        
        **Project Guidance:**
        - **Dr. Mahtab Uddin, Ph.D.**
        
        **Institution:**
        - Dhaka Residential Model College (DRMC)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Importance of DO")
        st.markdown("""
        Dissolved Oxygen (DO) is one of the most important indicators of water quality. It is essential for the survival of aquatic organisms and plays a crucial role in:
        
        - **Aquatic Life Support**: Most aquatic organisms need oxygen to survive
        - **Water Quality Assessment**: Low DO levels indicate pollution
        - **Ecosystem Health**: DO affects biodiversity and ecosystem stability
        - **Water Treatment**: DO levels influence water treatment processes
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Contact information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Contact Information")
    
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    
    with contact_col1:
        st.markdown("**Email**")
        st.markdown("contact@dopredict.example.com")
    
    with contact_col2:
        st.markdown("**Website**")
        st.markdown("www.dopredict.example.com")
    
    with contact_col3:
        st.markdown("**Institution**")
        st.markdown("Dhaka Residential Model College")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("© 2023 DO Prediction System | Developed by Mohammed Ahtesamul Rasul & Ilham Sajid | Guided by Dr. Mahtab Uddin")
st.markdown("</div>", unsafe_allow_html=True)