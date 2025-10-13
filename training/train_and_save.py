import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import requests

ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(ROOT, "dataset", "abalone.csv")
if not os.path.exists(DATA_PATH):
    print("Dataset not found locally. Downloading UCI Abalone dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    r = requests.get(url)
    if r.status_code == 200:
        raw = r.text
        header = "Sex,Length,Diameter,Height,WholeWeight,ShuckedWeight,VisceraWeight,ShellWeight,Rings\n"
        with open(DATA_PATH, "w") as fh:
            fh.write(header + raw)
        print("Downloaded and saved to", DATA_PATH)
    else:
        raise RuntimeError("Failed to download dataset. Please put abalone.csv in dataset/ folder.")

df = pd.read_csv(DATA_PATH)

# Basic preprocessing
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

X = df.drop('Rings', axis=1)
y = df['Rings']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_dt = dt.predict(X_test)
r2_dt = r2_score(y_test, y_dt)
mse_dt = mean_squared_error(y_test, y_dt)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_rf)
mse_rf = mean_squared_error(y_test, y_rf)

print("Decision Tree R2:", r2_dt, "MSE:", mse_dt)
print("Random Forest R2:", r2_rf, "MSE:", mse_rf)

# Choose best model by R2
best_model = dt if r2_dt >= r2_rf else rf
best_name = "DecisionTree" if best_model is dt else "RandomForest"
print("Saving best model:", best_name)

with open(os.path.join(ROOT, "abalone.pkl"), "wb") as f:
    pickle.dump(best_model, f)

print("Saved abalone.pkl at project root.")
