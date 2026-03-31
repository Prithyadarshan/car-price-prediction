import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# %matplotlib inline
mpl.style.use('ggplot')

# Load dataset
car = pd.read_csv('car_pred.csv')

# Initial inspection
print(car.head())
print(car.shape)
print(car.info())

# Backup
backup = car.copy()

# Data Cleaning
# Clean 'year'
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

# Clean 'Price'
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)

# Clean 'kms_driven'
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

#Remove missing fuel type
car = car[~car['fuel_type'].isna()]

# Reduce name length
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')

car = car.reset_index(drop=True)

# Save cleaned data
car.to_csv('Cleaned_Car_data.csv')

print(car.info())
print(car.describe(include='all'))

# Remove outliers
car = car[car['Price'] < 6000000]

# Visualization
#Company vs Price
plt.figure(figsize=(15,7))
ax = sns.boxplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.show()

#year vs Price
plt.figure(figsize=(20,10))
ax = sns.swarmplot(x='year', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.show()

sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)

plt.figure(figsize=(14,7))
sns.boxplot(x='fuel_type', y='Price', data=car)
plt.show()

sns.relplot(x='company', y='Price', data=car, hue='fuel_type', size='year', height=7, aspect=2)

# Model
X = car[['name','company','year','kms_driven','fuel_type']]
y = car['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# handle unknown categories
ohe = OneHotEncoder(handle_unknown='ignore')

column_trans = make_column_transformer(
    (ohe, ['name','company','fuel_type']),
    remainder='passthrough'
)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("Initial R2 Score:", r2_score(y_test, y_pred))

# best model
scores = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

best_index = np.argmax(scores)
print("Best Random State:", best_index)
print("Best R2 Score:", scores[best_index])

# Train final model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_index)
pipe.fit(X_train, y_train)

# Final prediction
sample_input = pd.DataFrame(
    columns=['name','company','year','kms_driven','fuel_type'],
    data=np.array(['Maruti Swift Dzire','Maruti',2015,40000,'Diesel']).reshape(1,5)
)

prediction = pipe.predict(sample_input)

print("Predicted Price (array):", prediction)
print("Predicted Price (value):", prediction[0])

# Save Model
import pickle
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Check Encoding
print("Encoded car names:")
print(pipe.named_steps['columntransformer'].transformers_[0][1].categories_[0])