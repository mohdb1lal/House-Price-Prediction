# California Housing Price Prediction

## Project Overview

This project aims to predict house prices in California using machine learning models. By analyzing various features such as location, housing characteristics, and demographic data, we develop a robust prediction model that can estimate median house values. The project showcases data preprocessing, exploratory data analysis, feature engineering, and the implementation of multiple machine learning algorithms.

## Key Objectives

- Analyze the relationship between various housing features and median house prices
- Develop accurate prediction models using different machine learning algorithms
- Identify the most significant factors influencing house prices in California

## Dataset Description

The dataset contains information about houses in California, including:

- Geographical information (longitude, latitude)
- Housing features (total rooms, bedrooms, population, households)
- Demographics (median income)
- Target variable: median house value
- Location category (ocean proximity)

### Sample Data

| longitude | latitude | housing_median_age | total_rooms | total_bedrooms | population | households | median_income | median_house_value | ocean_proximity |
| --------- | -------- | ------------------ | ----------- | -------------- | ---------- | ---------- | ------------- | ------------------ | --------------- |
| -122.23   | 37.88    | 41.0               | 880.0       | 129.0          | 322.0      | 126.0      | 8.3252        | 452600.0           | NEAR BAY        |
| -122.22   | 37.86    | 21.0               | 7099.0      | 1106.0         | 2401.0     | 1138.0     | 8.3014        | 358500.0           | NEAR BAY        |

## Project Structure

```
california_housing_prediction/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── california_housing_prediction.ipynb
│
├── images/
│   ├── distribution_before_after.png
│   ├── correlation_heatmap.png
│   ├── geographical_distribution.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```

## Setup and Installation

1. Clone the repository

```bash
git clone [your-repo-link]
```

2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run Jupyter Notebook

```bash
jupyter notebook notebooks/california_housing_prediction.ipynb
```

## Methodology

### 1. Data Preprocessing

- Handled missing values by dropping them
- Created train-test split (80-20 ratio)
- Applied log transformation to normalize distributions:
  - total_rooms
  - total_bedrooms
  - population
  - households

[IMAGE 1: Place distribution_before_after.png here showing histograms before and after log transformation]

### 2. Feature Engineering

- One-hot encoded the 'ocean_proximity' categorical variable
- Created new features:
  - bedroom_ratio (total_bedrooms / total_rooms)
  - household_rooms (total_rooms / households)

### 3. Exploratory Data Analysis

#### Correlation Analysis

[IMAGE 2: Place correlation_heatmap.png here showing the correlation between different features]

Key findings:

- Strong correlation between median_income and house value
- High correlation among household-related features

#### Geographical Distribution

[IMAGE 3: Place geographical_distribution.png here showing the scatter plot of latitude vs longitude, color-coded by house value]

The scatter plot reveals:

- Coastal areas tend to have higher house values
- Distinct clusters of high-value properties

### 4. Model Development

#### Model Comparison

| Model                         | R-squared Score |
| ----------------------------- | --------------- |
| Linear Regression             | 0.65            |
| Random Forest (before tuning) | 0.80            |
| Random Forest (after tuning)  | 0.82            |

#### Hyperparameter Tuning

Used GridSearchCV to optimize:

- n_estimators: [100, 200, 300]
- min_samples_split: [2, 4]
- max_depth: [None, 4, 8]

Best parameters:

```python
RandomForestRegressor(max_depth=None, min_samples_split=2, n_estimators=300)
```

#### Feature Importance

[IMAGE 4: Place feature_importance.png here showing the bar plot of feature importance]

## Results and Insights

1. The Random Forest model significantly outperformed Linear Regression
2. Most important features for prediction:
   - Median income
   - Location (latitude/longitude)
   - Household-related features
3. Log transformation of features improved model performance

## Future Improvements

- Experiment with other algorithms (XGBoost, LightGBM)
- Implement feature selection techniques
- Use cross-validation for more robust evaluation
- Collect additional features such as crime rate, school ratings

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
