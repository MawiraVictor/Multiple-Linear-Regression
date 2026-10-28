
# Multiple Linear Regression Model

This project implements a Multiple Linear Regression model to predict CO2 emissions based on two vehicle features: engine size and fuel consumption. It demonstrates the complete machine learning workflow, from data loading and preprocessing to model training and 3D visualization.

## 1. Setup and Data Loading

First, the required libraries (`numpy`, `pandas`, `scikit-learn`, `matplotlib`) are installed and imported.

The dataset is loaded from the IBM Skills Network cloud storage:
```python
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)
```

A random sample of 50 rows is displayed using `df.sample(50)` to verify the successful load.

## 2. Feature Selection and Preprocessing

To focus the model on the most relevant continuous variables, we perform the following steps:

1.  **Drop Categorical Columns:** Remove columns with text data (e.g., `MAKE`, `MODEL`, `VEHICLECLASS`).
2.  **Drop Redundant Features:** Remove highly correlated or less useful columns (e.g., `CYLINDERS`, `FUELCONSUMPTION_CITY`, `FUELCONSUMPTION_HWY`).
3.  **Select Final Features:** The final dataset contains two input features and one target:
    *   `ENGINESIZE`
    *   `FUELCONSUMPTION_COMB_MPG`
    *   `CO2EMISSIONS` (Target)

A scatter matrix is generated to visualize the relationships between these selected features.

## 3. Data Preparation and Scaling

The selected features are extracted into a NumPy array `X`, and the target is extracted into `y`. 

We apply **Standard Scaling** to the input features to standardize them (mean = 0, standard deviation = 1). This is a crucial step for linear regression to ensure that features with different scales do not disproportionately influence the model.

```python
from sklearn import preprocessing
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)
```

The data is then split into training (80%) and testing (20%) sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)
```

## 4. Model Training

A Multiple Linear Regression model is trained using `scikit-learn`:
```python
from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
```

The model learns the coefficients and intercept from the scaled data. We can then convert these coefficients back to the original, unstandardized feature space to see their true impact:
```python
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)
```

*Output:*
```
Coefficients:  [[17.8581369  -5.01502179]]
Intercept:  [329.1363967]
```
*This means that for every 1-unit increase in Engine Size, CO2 emissions increase by ~17.86 units, assuming fuel consumption remains constant.*

## 5. Visualization

### 3D Regression Plane
A 3D plot is generated to visualize the regression plane fitted to the data. The plot differentiates data points based on whether they are above or below the predicted plane, providing a clear visual representation of the model's fit in a multi-dimensional space.
<img width="485" height="359" alt="image" src="https://github.com/user-attachments/assets/e4716296-33b1-45fb-bc36-37bbd4385496" />

### 2D Feature Plots
Simplified 2D plots show the relationship between each individual feature and the target variable, along with the regression line.

*   **Engine Size vs. Emissions**
*   <img width="470" height="245" alt="image" src="https://github.com/user-attachments/assets/e735e283-4c31-4e31-8f12-a8039e04586d" />

*   **Fuel Consumption (MPG) vs. Emissions**
*   <img width="632" height="259" alt="image" src="https://github.com/user-attachments/assets/ba637092-5bf8-4081-88e9-7c64e809e154" />

```
