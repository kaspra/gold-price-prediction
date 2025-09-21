# Gold Price Prediction Model using Random Forest Regression

## Overview
This project implements a machine learning model to predict gold prices (GLD) using various financial indicators. The model leverages Random Forest Regression to forecast gold prices based on historical market data from multiple asset classes.

## Dataset
The model uses historical financial data spanning from **January 2008 to May 2018** with the following features:

- **SPX**: S&P 500 Index prices
- **GLD**: SPDR Gold Shares ETF (Target variable)
- **USO**: United States Oil Fund ETF
- **SLV**: iShares Silver Trust ETF  
- **EUR/USD**: Euro to US Dollar exchange rate

**Dataset Statistics:**
- Total records: 2,290 data points
- Time period: ~10.5 years
- No missing values
- Data source: Financial market data

## Model Architecture

### Algorithm: Random Forest Regressor
- **Estimators**: 100 decision trees
- **Train/Test Split**: 80/20 ratio
- **Random State**: 6


## Model Performance

### Evaluation Metrics
- **R² Score**: 0.9875 (98.75% accuracy)
- **Model Type**: Regression
- **Prediction Accuracy**: Excellent fit with actual vs predicted values

### Performance Visualization
The model shows strong predictive capability with predicted values closely following actual gold price movements, as demonstrated in the comparison plot.

## Installation & Requirements

```bash
# Required Python packages
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Data visualization
- `seaborn`: Statistical data visualization
- `scikit-learn`: Machine learning algorithms

## Usage

### 1. Data Loading
```python
gold_data = pd.read_csv('dataset/gold_price_data.csv')
```

### 2. Data Preprocessing
```python
# Feature selection
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=6)
```

### 3. Model Training
```python
# Initialize and train Random Forest model
regression = RandomForestRegressor(n_estimators=100)
regression.fit(X_train, Y_train)
```

### 4. Prediction & Evaluation
```python
# Make predictions
predictions = regression.predict(X_test)

# Evaluate performance
r2_score = metrics.r2_score(Y_test, predictions)
```

## File Structure
```
gold-price-prediction/
├── dataset/
│   └── gold_price_data.csv
├── gold-price-prediction.ipynb
└── README.md
```

## Key Insights

1. **Silver correlation**: The strongest predictor of gold prices is silver (SLV) with a correlation of 0.867
2. **Market independence**: Gold shows relative independence from stock market performance (SPX correlation: 0.049)
3. **Currency relationship**: Weak negative correlation with EUR/USD suggests gold's role as a currency hedge
4. **Oil relationship**: Slight negative correlation with oil prices


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## Disclaimer
This model is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with financial advisors before making investment choices.

