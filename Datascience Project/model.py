import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Reading Each File
country_wise_data_rank = pd.read_csv("cwurData.csv")

# Displaying basic information about the dataset
print(country_wise_data_rank.info())

# Sorting the index
print(country_wise_data_rank.sort_index())

# Displaying the shape of the dataset
print("cwdr: ", country_wise_data_rank.shape)

# Descriptive statistics
print(country_wise_data_rank.describe())

# Displaying the first 5 rows
print(country_wise_data_rank.head())

# Displaying the last 5 rows
print(country_wise_data_rank.tail())

# Grouping data by 'Country' and calculating the average publication of each country
temp1 = country_wise_data_rank.groupby('country').publications.mean()
print(temp1)

# Median of publications
print("Median of publications:", country_wise_data_rank['publications'].median())

# Mode of publications
print("Mode of publications:", country_wise_data_rank['publications'].mode())

# Checking for missing values
print(country_wise_data_rank.isnull().sum())

# Filling missing values in 'broad_impact' with the mean value
print(country_wise_data_rank['broad_impact'].value_counts().sort_values())
country_wise_data_rank['broad_impact'] = country_wise_data_rank['broad_impact'].fillna(country_wise_data_rank['broad_impact'].mean())
print(country_wise_data_rank.isnull().sum())

# Checking for duplicate rows
print("Number of duplicate rows:", country_wise_data_rank.duplicated().sum())

# Dropping duplicate rows
country_wise_data_rank.drop_duplicates(inplace=True)
print("Shape after dropping duplicates:", country_wise_data_rank.shape)

# Displaying unique values in different columns
print("Unique values in world_rank:", country_wise_data_rank['world_rank'].nunique())
print("Unique values in institution:", country_wise_data_rank['institution'].nunique())
print("Unique values in country:", country_wise_data_rank['country'].nunique())
print("Unique values in national_rank:", country_wise_data_rank['national_rank'].nunique())
print("Unique values in quality_of_education:", country_wise_data_rank['quality_of_education'].nunique())
print("Unique values in alumni_employment:", country_wise_data_rank['alumni_employment'].nunique())

# Reading another CSV file
td = pd.read_csv("timesData.csv")

# Pie chart for year-wise ranking in Times data
temp1 = td.year.value_counts()
labels = td.year.value_counts().index
pl.pie(temp1, labels=labels, autopct='%.2f %%')
pl.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
pl.title('Year wise Ranking in Times data')
pl.show()

# Bar plot for top 20 countries with the most ranked universities
country_counts = country_wise_data_rank['country'].value_counts()
top_countries = country_counts.head(20)
fig, ax = pl.subplots(figsize=(12, 8))
top_countries.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Top 20 Countries with Most Ranked Universities')
ax.set_xlabel('Country')
ax.set_ylabel('Number of Universities')
pl.xticks(rotation=45)
pl.show()

# Bar plot for top 10 universities with the highest alumni employment rates
fig, ax = pl.subplots(figsize=(10, 6))
top10 = country_wise_data_rank.sort_values(by='alumni_employment', ascending=False).head(10)
top10.plot(kind='bar', x='institution', y='alumni_employment', color='green', ax=ax)
ax.set_title('Top 10 Universities with the Highest Alumni Employment Rates')
ax.set_xlabel('University')
ax.set_ylabel('Alumni Employment Rate')
pl.xticks(rotation=45)
pl.show()

# Bar plot for top 10 institutions in the USA based on national ranking
usa_institutions = country_wise_data_rank[country_wise_data_rank['country'] == 'USA']
top_usa_institutions = usa_institutions.head(10)
fig, ax = pl.subplots(figsize=(12, 8))
bars = ax.bar(top_usa_institutions['institution'], top_usa_institutions['national_rank'], color='blue')
ax.set_title('Top 10 Institutions in the USA Based on National Ranking')
ax.set_xlabel('Institution')
ax.set_ylabel('National Ranking')
pl.xticks(rotation=45)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
pl.show()

# Histogram for the distribution of publications
pl.figure(figsize=(10, 6))
pl.hist(country_wise_data_rank['publications'], bins=20, color='skyblue', edgecolor='black')
pl.title('Distribution of Publications')
pl.xlabel('Number of Publications')
pl.ylabel('Frequency')
pl.show()

# Random Forest algorithm for predicting world_rank
# Feature selection
features = ['publications', 'quality_of_education', 'alumni_employment']
X = country_wise_data_rank[features]
y = country_wise_data_rank['world_rank']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Random Forest Mean Squared Error:", mse_rf)

# Plotting feature importances from the Random Forest model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]

pl.figure(figsize=(10, 6))
pl.title("Feature Importances from Random Forest")
pl.bar(range(X.shape[1]), importances[indices], align='center')
pl.xticks(range(X.shape[1]), feature_names, rotation=45)
pl.xlabel('Feature')
pl.ylabel('Importance')
pl.show()

# Message on model performance
print("\nThe Random Forest model has a Mean Squared Error of:", mse_rf)
print("The feature importances plot shows which features are most influential in predicting the world ranking.")
