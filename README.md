📈 TDA-Enhanced Stock Price Direction Prediction using Persistent Homology & XGBoost

This project explores Topological Data Analysis (TDA) in the context of financial time series by leveraging persistent homology to extract shape-based features from stock price data. These features are then fed into a machine learning model (XGBoost) to predict the direction of future stock price movement.

🧠 Core Concepts
🔹 Topological Data Analysis (TDA)

Topological Data Analysis (TDA) is a set of techniques from algebraic topology used to analyze the shape of data. In this project, TDA is used to analyze the geometric and topological structures present in sliding windows of stock price time series.

✅ Persistent Homology

We compute persistent homology using the ripser
 library. Persistent homology tracks topological features (e.g., connected components, loops) across different scales:

H0: Connected components

H1: Loops/cycles (used here)

From the resulting persistence diagrams, we extract summary statistics such as:

Sum of lifetimes

Maximum lifetime

Mean lifetime

These features represent the "shape" of the time series window and are used as input to the machine learning model.

🔹 Time Delay Embedding

To uncover latent geometric structures, we convert each window of 1D stock prices into a higher-dimensional space using time-delay embedding:

𝑥
𝑡
→
[
𝑥
𝑡
,
𝑥
𝑡
+
𝜏
,
𝑥
𝑡
+
2
𝜏
,
.
.
.
,
𝑥
𝑡
+
(
𝑑
−
1
)
𝜏
]
x
t
	​

→[x
t
	​

,x
t+τ
	​

,x
t+2τ
	​

,...,x
t+(d−1)τ
	​

]

Where:

𝑑
d: embedding dimension

𝜏
τ: time delay

This allows us to reveal topological patterns that may indicate temporal dynamics.

🔹 Machine Learning: XGBoost Classifier

We use XGBoost, a high-performance gradient-boosted decision tree algorithm, to learn directional changes in price movement:

Input: Topological features (from persistent homology)

Target: Binary classification (1 if price goes up, 0 otherwise)

Training is done using a train-test split, and model performance is measured using accuracy.

🚀 How It Works
1. Load and Preprocess Data

Dataset: all_stocks_5yr.csv

First numeric column (open) is used

NaNs are interpolated

2. Sliding Window & Feature Extraction

Slide over the time series with a fixed window and stride

For each window:

Apply time-delay embedding

Compute persistent homology (maxdim = 1)

Extract lifetimes and derive summary features

3. Prepare Labels

Predict direction: whether price increased in the step after the window

4. Train Model

Use XGBoost classifier

Evaluate on the test set (non-shuffled to preserve temporal order)

📊 Results

Accuracy: ~49.3% (varies by run and dataset length)

Visual comparison of actual vs predicted directions is plotted

⚠️ Note: Accuracy close to 50% indicates performance similar to random guessing. Further tuning, richer features, or more complex models may be required.

🧪 Example Output
Using numeric column: open  (length=619040)
Features shape: (12377, 6)
Targets shape: (12377,)
Class distribution: (array([0, 1]), array([5849, 6528], dtype=int64))
✅ XGBoost Accuracy: 0.4927


🛠 Dependencies

Python 3.7+

pandas

numpy

matplotlib

ripser

scikit-learn

xgboost

Install with:

pip install -r requirements.txt

📂 Project Structure
├── all_stocks_5yr.csv     # Input CSV file
├── tda_forecasting.ipynb  # Main notebook
├── README.md              # This file
└── requirements.txt       # Python dependencies

📌 Limitations & Future Work

Current model uses only basic TDA features; additional features (e.g., entropy, bottleneck distances) may help.

Incorporate multiple time series or technical indicators.

Try other ML models or deep learning (e.g., LSTM, transformers).

Perform cross-validation and hyperparameter tuning.

📚 References

G. Carlsson, "Topology and Data", Bulletin of the AMS, 2009.

Scikit-TDA: https://scikit-tda.org/

Ripser: https://github.com/scikit-tda/ripser.py

XGBoost: https://xgboost.readthedocs.io
