Project Overview

This project aims to predict the price of used cars using machine learning techniques. It involves data cleaning, preprocessing and feature transformation to handle real-world inconsistencies in the dataset. The system helps estimate car prices based on factors such as brand, year, fuel type and kilometers driven, supporting better decision making for buyers and sellers.

Objectives

* Analyze car dataset to predict price accurately.
* Perform data cleaning and preprocessing.
* Build a regression model for price prediction.
* Evaluate model performance using suitable metrics.

Key Features

* Car price prediction using Linear Regression model.
* Data preprocessing including cleaning inconsistent values.
* Handling categorical variables using One Hot Encoding.
* Pipeline implementation for efficient model building.
* Model evaluation using R2 score.
* Visualization of relationships between features and price.

Technology Stack

* Python
* Pandas and NumPy for data handling
* Matplotlib and Seaborn for visualization
* Scikit learn for machine learning

Dataset Description

* Dataset contains used car information.
* Includes features such as name, company, year, price, fuel type and kilometers driven.
* Data contains categorical and numerical features.
* Some entries include inconsistent values like “Ask For Price” which are cleaned during preprocessing.

Data Preprocessing

* Removed non numeric values in year column and converted to integer.
* Cleaned price column by removing text and commas.
* Processed kilometers driven column to extract numeric values.
* Handled missing values in fuel type.
* Reduced car name length for better categorization.
* Removed outliers in price to improve model performance.

Exploratory Data Analysis

* Visualized price distribution across different companies.
* Analyzed relationship between year and price.
* Studied impact of kilometers driven on price.
* Compared price variations based on fuel type.

Model Evaluation

* R2 Score is used to evaluate model performance.
* Multiple random states are tested to find the best model.
* Best model is selected based on highest R2 score.

Expected Outcomes

* Accurate prediction of used car prices.
* Understanding of factors affecting car price.
* Improved decision making for buyers and sellers.

Results

* Model achieved a good R2 score indicating reliable predictions.
* Best random state is selected for optimal performance.

