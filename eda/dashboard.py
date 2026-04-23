"""
EDA Dashboard
─────────────────────
Streamlit app presenting EDA findings, model performance comparisons,
and business insights in a non-technical format for stakeholders.

Run:
    streamlit run eda/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eda.visualize import (
    plot_target_distribution,
    plot_price_by_city,
    plot_correlation_heatmap,
    plot_feature_vs_target,
    plot_geospatial_scatter,
    plot_numeric_distributions,
    plot_price_by_room_type,
    plot_reviews_vs_rating,
    plot_amenity_analysis
)

from feature_engineering.selection import categorize_rating

# Page configuration
st.set_page_config(
    page_title="Airbnb Rating Classification Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF5A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #484848;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(data_path: str = "data/processed/cleaned.csv"):
    """Load cleaned data with caching."""
    try:
        df = pd.read_csv(data_path)
        
        # Apply rating categorization with default thresholds (0-3, 3-4.51, 4.51-4.91, 4.91-5)
        df['rating_category'] = df['review_scores_rating'].apply(
            lambda x: categorize_rating(x, thresholds=(3.0, 4.51, 4.91))
        )
        
        # Reorder rating_category to desired order
        category_order = ['Low Rating', 'Medium Rating', 'High Rating', 'Very High Rating']
        # Only include categories that exist in the data
        existing_categories = [cat for cat in category_order if cat in df['rating_category'].unique()]
        df['rating_category'] = pd.Categorical(
            df['rating_category'],
            categories=existing_categories,
            ordered=True
        )
        
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {data_path}")
        return None


@st.cache_data
def load_featured_data(data_path: str = "data/processed/featured.csv"):
    """Load featured data with caching."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        return None


def show_eda_section(df):
    """Display EDA visualizations with interactive controls."""
    
    st.markdown('<p class="sub-header"> Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Key Business Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Listings", f"{len(df):,}")
    with col2:
        avg_rating = df['review_scores_rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}/5.0")
    with col3:
        avg_price = df['price'].mean()
        st.metric("Avg Price", f"${avg_price:.0f}/night")
    with col4:
        high_rated_pct = (df['review_scores_rating'] >= 4.51).sum() / len(df) * 100
        st.metric("High Rated", f"{high_rated_pct:.1f}%", help="Listings with 4.51+ rating")
    with col5:
        avg_capacity = df['accommodates'].mean()
        st.metric("Avg Capacity", f"{avg_capacity:.1f} guests")
    
    # Visualization selector
    st.markdown("---")
    st.markdown("### 🎨 Interactive Visualizations")
    
    # =============================================================================
    # 1. TARGET DISTRIBUTION
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Rating Distribution Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        show_binned = st.checkbox("Show binned categories", value=True, key="target_binned")
    
    with col2:
        if show_binned and 'rating_category' in df.columns:
            binned_col = 'rating_category'
        else:
            binned_col = None
    
    fig = plot_target_distribution(df, target_col='review_scores_rating', binned_col=binned_col)
    st.pyplot(fig)
    plt.close()
    
    # =============================================================================
    # 2. FEATURE DISTRIBUTION BY CATEGORY
    # =============================================================================
    st.markdown("---")
    st.markdown("#### 📈 Feature Distribution by Category")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        y_col = st.selectbox("Y-axis (Numeric):", numeric_cols, 
                            index=numeric_cols.index('price') if 'price' in numeric_cols else 0,
                            key="feature_y")
    
    with col2:
        categorical_cols = ['room_type', 'city']
        categorical_cols = [c for c in categorical_cols if c in df.columns]
        x_col = st.selectbox("X-axis (Category):", categorical_cols, key="feature_x")
    
    with col3:
        plot_type = st.selectbox("Plot Type:", ["box", "violin"], key="feature_plot_type")
    
    # Create custom plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if plot_type == "violin":
        sns.violinplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2")
    else:
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2")
    
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{y_col.replace("_", " ").title()} by {x_col.replace("_", " ").title()}', 
                fontsize=14, pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()
    
    # =============================================================================
    # 3. CORRELATION ANALYSIS
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Feature Correlation Heatmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox("Correlation Method:", ["pearson", "spearman", "kendall"], key="corr_method")
    with col2:
        show_annot = st.checkbox("Show Values", value=True, key="corr_annot")
    
    # Select features for correlation
    numeric_cols_corr = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Select Features (leave empty for all numeric):",
        numeric_cols_corr,
        default=[],
        key="corr_features"
    )
    
    features_to_plot = selected_features if selected_features else None
    
    fig = plot_correlation_heatmap(df, features=features_to_plot, method=method, 
                                  annot=show_annot)
    st.pyplot(fig)
    plt.close()
    
    # =============================================================================
    # 4. GEOGRAPHIC DISTRIBUTION
    # =============================================================================
    st.markdown("---")
    st.markdown("#### Geographic Distribution of Listings")
    
    # Selector for color by option
    color_option = st.selectbox(
        "Color by:",
        ["rating_category", "price"],
        key="geo_color"
    )
    
    # Use selected option for geographic visualization
    if 'latitude' in df.columns and 'longitude' in df.columns:
        fig = plot_geospatial_scatter(
            df,
            lat_col='latitude',
            lon_col='longitude',
            color_by=color_option,
            city_col='city'
        )
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Geographic coordinates not available in this dataset.")


def show_predict_section():
    """Display prediction demo interface."""
    
    st.markdown('<p class="sub-header">🎯 Rating Prediction Demo</p>', unsafe_allow_html=True)
    
    st.info("💡 **Interactive Model Demo** - Enter listing details to predict rating category")
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("### Listing Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Location**")
            city = st.selectbox("City:", ["New York", "Los Angeles", "San Francisco", 
                                         "Washington DC", "Chicago", "Boston"])
            neighbourhood = st.text_input("Neighbourhood:", value="Downtown")
            
            st.markdown("**Property**")
            property_type = st.selectbox("Property Type:", ["Apartment", "House", "Condo", 
                                                           "Townhouse", "Loft"])
            room_type = st.selectbox("Room Type:", ["Entire home/apt", "Private room", "Shared room"])
        
        with col2:
            st.markdown("**Capacity**")
            accommodates = st.number_input("Accommodates:", min_value=1, max_value=16, value=2)
            bedrooms = st.number_input("Bedrooms:", min_value=0, max_value=10, value=1)
            bathrooms = st.number_input("Bathrooms:", min_value=0, max_value=10, value=1)
            
            st.markdown("**Host**")
            host_verified = st.checkbox("Host Identity Verified", value=True)
            host_response_rate = st.slider("Host Response Rate (%):", 0, 100, 95)
        
        with col3:
            st.markdown("**Pricing & Amenities**")
            price = st.number_input("Price per Night ($):", min_value=10, max_value=2800, value=100)
            amenity_count = st.number_input("Number of Amenities:", min_value=0, max_value=100, value=20)
            
            st.markdown("**Reviews**")
            number_of_reviews = st.number_input("Total Reviews:", min_value=0, max_value=500, value=10)
        
        # Submit button
        submitted = st.form_submit_button("Predict Rating", width=True)
    
    if submitted:
        # Create feature dictionary
        features = {
            'city': city,
            'neighbourhood': neighbourhood,
            'property_type': property_type,
            'room_type': room_type,
            'accommodates': accommodates,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'host_identity_verified': host_verified,
            'host_response_rate': host_response_rate,
            'price': price,
            'amenity_count': amenity_count,
            'number_of_reviews': number_of_reviews
        }
        
        # TODO: Load actual model and make prediction
        # For now, show placeholder
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Predicted Rating")
            # Placeholder prediction logic
            if price > 150 and amenity_count > 25:
                predicted_category = "Very High Rating"
                predicted_score = 4.85
            elif price < 75 or amenity_count < 10:
                predicted_category = "Medium Rating"
                predicted_score = 4.2
            else:
                predicted_category = "High Rating"
                predicted_score = 4.65
            
            st.metric("Category", predicted_category)
            st.metric("Estimated Score", f"{predicted_score:.2f} / 5.0")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Model Confidence")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### Class Probabilities")
            st.write("Medium Rating: 15%")
            st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">Airbnb Rating Classification Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("**Applied Data Science Project** - Spring 2026")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select View:",
        ["📊 EDA - Exploratory Analysis", "Predict - Model Demo"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Load cleaned data only
    df = load_data("data/processed/cleaned.csv")
    
    if df is None:
        st.error("Failed to load data. Please check that data files exist in data/processed/")
        st.stop()
    
    # Sidebar info
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"**Rows:** {len(df):,}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")
    st.sidebar.write(f"**Cities:** {df['city'].nunique() if 'city' in df.columns else 'N/A'}")
    st.sidebar.write(f"**Avg Rating:** {df['review_scores_rating'].mean():.2f}")
    st.sidebar.write(f"**Price Range:** ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    
    # Main content based on selection
    if "EDA" in page:
        show_eda_section(df)
    else:
        show_predict_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Airbnb Rating Classification Dashboard | Built with Streamlit</p>
        <p>Applied Data Science - Spring 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
