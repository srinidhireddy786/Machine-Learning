import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config("Multiple Linear Regression", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style1.css")

# Title
st.markdown("""
<div class="card">
<h1>Multiple Linear Regression</h1>
<p>Predict <b>Tip Amount</b> using multiple features</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Dataset preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Data Preparation
# -----------------------------
X = df.drop("tip", axis=1)
y = df["tip"]

# One-hot encoding categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# -----------------------------
# Visualization (Actual vs Predicted)
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Actual vs Predicted Tip")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red")
ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Performance
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Coefficients
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Coefficients")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

st.dataframe(coef_df)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider("Total Bill ($)", float(df.total_bill.min()), float(df.total_bill.max()), 30.0)
size = st.slider("Party Size", 1, 6, 2)

sex = st.selectbox("Gender", df["sex"].unique())
day = st.selectbox("Day", df["day"].unique())
time = st.selectbox("Time", df["time"].unique())

input_data = pd.DataFrame({
    "total_bill": [bill],
    "size": [size],
    "Gender": [sex],
    "day": [day],
    "time": [time]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${prediction:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
