import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

housing_df = pd.read_csv('housing_data.csv')
print("==================================")
print(housing_df.head())
print("==================================")
print(housing_df.info())
print("==================================")
print(housing_df['TAX'].value_counts())
print("==================================")
print(housing_df.describe())
print("==================================")


housing_df.hist(bins=50 , figsize=(20,15))

# For Info purpose
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print("Shuffled", shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing_df, 0.2)

# print(f"Train data rows: {len(train_set)}")
# print(f"Test data rows: {len(test_set)}")

train_set, test_set = train_test_split(housing_df, test_size = 0.2, random_state =42)
# print(f"Train data rows: {len(train_set)}")
# print(f"Test data rows: {len(test_set)}")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing_df , housing_df['CHAS']):
    strate_train_set = housing_df.loc[train_index]
    strate_test_set = housing_df.loc[test_index]

# print("Strate train set :", strate_train_set)
# print("Strate test set :", strate_test_set)

print(strate_train_set['CHAS'].value_counts())
print("==================================")
print(strate_test_set['CHAS'].value_counts())
print("==================================")

housing_df = strate_train_set.copy()

corr_matrix = housing_df.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing_df[attributes], figsize=(10,8))


housing_df.plot(kind="scatter", x="RM", y="MEDV", alpha = 0.8)


# trying attribute combinations
housing_df["TAXRM"] = housing_df["TAX"]/housing_df["RM"]
print(housing_df.head())
print("==================================")

corr_matrix = housing_df.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing_df.plot(kind="scatter", x="TAXRM", y="MEDV", alpha = 0.8)
plt.show()

housing  = strate_train_set.drop("MEDV", axis =1)
housing_labels = strate_train_set["MEDV"].copy()

#  missing value|
# (1) get rid of missing data points 
# a = housing_df.dropna(subset=["RM"])
# a.shape

# (2) get rid of whole attribute
# a = housing_df.drop("RM", axis=1)
# a

# (3) set the value to 0 or mean or median
# median = housing_df["RM"].median()
# median
# housing_df["RM"].fillna(median)

print(housing_df.describe())
print("==================================")
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_df)

print(imputer.statistics_)
print("==================================")

x = imputer.transform(housing_df)
housing_df_2 = pd.DataFrame(x, columns = housing_df.columns)
housing_df_2.describe()

#  pipeline 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy="median")),  
  ('std_scaler', StandardScaler()),
])

housing_num_df_2 = pipeline.fit_transform(housing)
print(housing_num_df_2)
print("==================================")

# model selection

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_df_2, housing_labels)

data = housing.iloc[:5]
data_lebels = housing_labels.iloc[:5]
prepared_data = pipeline.transform(data)
model.predict(prepared_data)

list(data_lebels)

housing_predictions = model.predict(housing_num_df_2)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(lin_mse)
print(lin_mse)
print("==================================")
print(rmse)
print("==================================")

# cross validation evaluation

scores = cross_val_score(model , housing_num_df_2, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print("==================================")

def get_scores(scores):
    print("==================================")
    print("==================================")
    print("scores : ", scores)
    print("Mean : ", scores.mean())
    print("Standard deviation : ", scores.std())
    print("==================================")
    print("==================================")

get_scores(rmse_scores)

# Dumb model
from joblib import dump, load
dump(model, 'model.joblib')


# Testing the model
X_test = strate_test_set.drop("MEDV", axis=1)
Y_test = strate_test_set["MEDV"].copy()
X_test_prepared = pipeline.transform(X_test)

final_predictions = model.predict(X_test_prepared)

final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
print("==================================")
print(prepared_data[0])
print("==================================")

#  use the model
from joblib import load
import numpy as np
model = load('model.joblib')

input_data = np.array([[2.44499072,  8.12628155, 1.12165014, -0.27288841, -1.42262747,
       -0.24204419, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -9.86091034]])
model.predict(input_data)





