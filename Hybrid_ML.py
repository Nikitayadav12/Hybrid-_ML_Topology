#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ripser gudhi xgboost matplotlib pandas scikit-learn')


# In[3]:


get_ipython().system('pip install "blosc2~=2.0.0"')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ripser import Rips
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# In[5]:


# Example: Load from local CSV (replace filename if needed)
df = pd.read_csv("all_stocks_5yr.csv")

# Pick first numeric column
numeric_cols = df.select_dtypes(include=np.number).columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric column found in CSV!")

prices = df[numeric_cols[0]].interpolate().dropna().values
print(f"Using numeric column: {numeric_cols[0]}  (length={len(prices)})")

plt.figure(figsize=(10,4))
plt.plot(prices)
plt.title("Price Series")
plt.show()


# In[6]:


def time_delay_embedding(series, dim=5, tau=1):
    n = len(series) - (dim - 1) * tau
    return np.array([series[i:i + dim * tau:tau] for i in range(n)])


# In[7]:


rips = Rips(maxdim=1, verbose=False)

window_size = 200   # number of points per window
step = 50           # stride

features, targets = [], []

for start in range(0, len(prices) - window_size, step):
    window = prices[start:start + window_size]
    embedded = time_delay_embedding(window, dim=5, tau=1)

    try:
        diagrams = rips.fit_transform(embedded)
    except Exception as e:
        print(f"Skipping window {start} due to error: {e}")
        continue

    # Feature extraction (lifetimes)
    f_vec = []
    for pdgm in diagrams:
        if pdgm.shape[1] == 2:
            lifetimes = pdgm[:, 1] - pdgm[:, 0]
            if len(lifetimes) == 0:
                lifetimes = np.array([0])
            f_vec += [lifetimes.sum(), lifetimes.max(), lifetimes.mean()]
    features.append(f_vec)

    # Target: direction of next step after window
    if start + window_size < len(prices) - 1:
        targets.append(int(prices[start + window_size] > prices[start + window_size - 1]))

features = np.array(features)
targets = np.array(targets)

print("Features shape:", features.shape)
print("Targets shape:", targets.shape)
print("Class distribution:", np.unique(targets, return_counts=True))


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ------------------------------
# Step 5.5: Sanitize features
# ------------------------------
features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

# ------------------------------
# Step 6: ML model
# ------------------------------
if len(np.unique(targets)) < 2:
    print("⚠️ Not enough class variety in targets. Try longer dataset.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(eval_metric="logloss", missing=0.0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("✅ XGBoost Accuracy:", acc)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("TDA + XGBoost Direction Forecast")
    plt.show()


# In[ ]:




