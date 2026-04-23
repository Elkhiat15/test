# EDA Dashboard

Interactive Streamlit dashboard for exploring Airbnb data and testing the rating prediction model.

## Features

### 📊 EDA Section

Explore the data with interactive visualizations:

1. **Target Distribution** - View rating distributions (continuous and binned)
2. **Feature Distribution by Category** - Analyze any numeric feature by categorical groups
   - Customizable X and Y axes
   - Box plot or violin plot options
   - Statistics table included
3. **Correlation Analysis** - Interactive correlation heatmap
   - Choose correlation method (Pearson, Spearman, Kendall)
   - Select specific features or view all
   - Adjustable threshold for highlighting strong correlations
4. **Geographic Distribution** - Map view of listings
   - Color by rating, price, rating category, or room type
   - Optional percentile scaling for price
5. **Missing Values Analysis** - Identify data quality issues
6. **Numeric Feature Distributions** - Histograms with KDE overlays
7. **Amenity Analysis** - Relationship between amenities, rating, and price

### 🎯 Predict Section

Demo interface for model predictions:

- **Input Form** - Enter listing details:
  - Location (city, neighbourhood)
  - Property details (type, room type, capacity)
  - Host information (verification, response rate)
  - Pricing and amenities
  - Review count
- **Prediction Results** - Shows:
  - Predicted rating category
  - Estimated score (0-5)
  - Model confidence
  - Class probabilities
- **Recommendations** - Actionable suggestions to improve rating
- **Note:** Currently uses placeholder logic. Will be updated with trained model.

## Installation

```bash
# Install Streamlit (if not already installed)
pip install streamlit

# Or install all requirements
pip install -r requirements.txt
```

## Usage

### Start the Dashboard

From the project root directory:

```bash
streamlit run eda/dashboard.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Navigation

**Sidebar Options:**

1. **View Selection:**
   - 📊 EDA - Exploratory Analysis
   - 🎯 Predict - Model Demo

2. **Data Source:**
   - Cleaned Data (`data/processed/cleaned.csv`)
   - Featured Data (`data/processed/featured.csv`)

### Example Use Cases

**Scenario 1: Explore Price by Room Type**
1. Select "EDA - Exploratory Analysis"
2. Choose "Feature Distribution by Category"
3. Set Y-axis to "price"
4. Set X-axis to "room_type"
5. Choose "box" or "violin" plot
6. View statistics table below the plot

**Scenario 2: Check Rating by Amenities**
1. Select "Feature Distribution by Category"
2. Set Y-axis to "review_scores_rating"
3. Create amenity bins or use amenity_count (if available)
4. Analyze the relationship

**Scenario 3: Test Model Prediction**
1. Select "Predict - Model Demo"
2. Fill in listing details in the form
3. Click "Predict Rating"
4. View predicted category, confidence, and recommendations

## Customization

### Add New Visualizations

To add custom plots, edit `dashboard.py`:

```python
# In show_eda_section()
elif viz_option == "Your New Viz":
    # Your custom plotting code
    fig, ax = plt.subplots()
    # ... create your plot
    st.pyplot(fig)
    plt.close()
```

### Modify Prediction Logic

Once the model is trained, update the prediction section:

```python
# In show_predict_section()
# Replace placeholder logic with:
import joblib
model = joblib.load('models/best_model.pkl')
prediction = model.predict(features_df)
```

## File Structure

```
eda/
├── dashboard.py          # Main Streamlit application
├── visualize.py          # Reusable plotting functions
├── EDA_GUIDE.md         # EDA findings documentation
├── eda.ipynb            # Jupyter exploration notebook
└── README.md            # This file
```

## Troubleshooting

### "Data file not found" Error

Make sure you're running from the project root directory and that data files exist:

```bash
cd /path/to/Project/DS
ls data/processed/cleaned.csv
streamlit run eda/dashboard.py
```

### Plot Display Issues

If plots don't appear:
- Refresh the browser
- Check terminal for errors
- Ensure matplotlib backend is set correctly

### Memory Issues with Large Datasets

For large datasets, Streamlit caches data automatically. To clear cache:
- Press 'C' in the terminal running Streamlit
- Or click "Clear cache" in the hamburger menu (top-right)

## Performance Tips

- Use data sampling for large datasets (>100K rows)
- Cache expensive computations with `@st.cache_data`
- Limit number of plots shown simultaneously
- Use expanders to hide detailed information

## Future Enhancements

- [ ] Add model comparison section (MLflow integration)
- [ ] Export functionality for plots
- [ ] Real-time model predictions with confidence intervals
- [ ] Feature importance visualization
- [ ] A/B testing simulator for hosts
- [ ] Download predictions as CSV

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (for future model integration)

See `requirements.txt` for versions.

## Contributing

To add new features:
1. Create visualizations in `visualize.py`
2. Import and use in `dashboard.py`
3. Add docstrings and type hints
4. Test with both cleaned and featured datasets

## License

Part of the Airbnb Rating Classification project - Spring 2026
