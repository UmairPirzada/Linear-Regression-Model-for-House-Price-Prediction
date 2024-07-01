# Linear Regression Model for House Price Prediction

This repository contains the implementation of a linear regression model to predict house prices based on their square footage, number of bedrooms, and number of bathrooms.
# Dataset: 
The dataset used for this project is from Kaggle: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset.


# Project Structure
### PRODIGY_ML_01.ipynb: 
Jupyter notebook containing the complete implementation of the linear regression model.
### README.md: 
Project description and instructions.
# Dependencies
The project requires the following Python libraries:

numpy

pandas

scikit-learn

matplotlib

seaborn

# Implementation Steps

### Import Required Libraries: 
Import essential libraries for data manipulation, visualization, and machine learning.
### Load and Explore Data:
Load the dataset and perform initial exploration to understand its structure.
### Data Cleaning:
Handle missing or infinite values in the dataset.
### Data Visualization: 
Visualize the data to understand the relationships between different features.
### Feature Selection:
Select relevant features for the model.
### Create Independent and Dependent Variables:
Define the feature matrix X and the target vector y.
### Split the Dataset:
Split the dataset into training and testing sets.
### Model Training:
Train a linear regression model using the training set.
### Model Prediction:
Make predictions on the test set.
### Model Evaluation:
Evaluate the model using Mean Squared Error (MSE) and R-Squared (R²) metrics.
### Results Visualization:
Visualize the actual vs predicted prices to understand the model's performance.

### 1. Import Required Libraries
The notebook starts by importing essential libraries for data manipulation, visualization, and machine learning:

numpy and pandas for data manipulation

train_test_split and LinearRegression from sklearn for model training

mean_squared_error and r2_score from sklearn for model evaluation

seaborn and matplotlib for data visualization

### 2. Load and Explore Data
The dataset is loaded from a CSV file into a pandas DataFrame.

Initial data exploration includes displaying the first few rows and checking the shape of the dataset.

Statistical summary of the data using the describe() function.

### 3. Data Cleaning

Infinite values are replaced with NaN, and rows with NaN values in the 'price' column are dropped.

### 4. Data Visualization
Scatter plot of Price vs Area, colored by the number of bedrooms.
Count plots for the number of bedrooms and bathrooms.
### 5. Feature Selection
Selected relevant columns:
'price', 'area', 'bedrooms', and 'bathrooms'.

### 6. Creating Independent and Dependent Variables
X contains the features 'area', 'bedrooms', and 'bathrooms'.

y contains the target variable 'price'.
### 7. Splitting the Dataset
The dataset is split into training and testing sets using an 80-20 split.
### 8. Model Training
Initialized and trained a linear regression model using the training set.
### 9. Model Prediction
Made predictions on the test set.
### 10. Model Evaluation
Calculated Mean Squared Error (MSE) and

R-Squared (R²) to evaluate the model's performance.
### 11. Results Visualization
Scatter plot comparing actual prices vs predicted prices.
# Results
### Mean Squared Error (MSE): 
The notebook computes and prints the MSE.
### R-Squared (R²): 
The notebook computes and prints the R² value.
# Visualizations
### Price vs Area:
A scatter plot showing the relationship between price and area, with points colored by the number of bedrooms.
### Actual vs Predicted Prices: 
A scatter plot comparing actual prices to predicted prices, showing how well the model performs.
# Conclusion
The notebook successfully implements a linear regression model to predict house prices based on their square footage, the number of bedrooms, and the number of bathrooms. The results are evaluated using MSE and R², and visualizations are provided to understand the model's performance. 

# Usage
To run the project, follow these steps:

# Clone the repository:
git clone https://github.com/UmairPirzada/PRODIGY_ML_01.git

# Open the Jupyter notebook:

jupyter notebook PRODIGY_ML_01.ipynb

# Thank You!
