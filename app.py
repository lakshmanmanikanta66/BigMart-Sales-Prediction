import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Load Model
# ------------------------------
pipeline = joblib.load("retail_full_pipeline.pkl")

# ------------------------------
# Load Data
# ------------------------------
train = pd.read_csv("train_.csv")

# ------------------------------
# Cleaning (same as notebook)
# ------------------------------
num_cols = train.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    train[col].fillna(train[col].median(), inplace=True)

cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)

# Feature Engineering
train["Outlet_Age"] = 2024 - train["Outlet_Establishment_Year"]
train.drop("Outlet_Establishment_Year", axis=1, inplace=True)

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🛒 Retail Sales Intelligence Dashboard")
st.write("Predict sales and perform inventory optimization.")

# ------------------------------
# PREDICT SALES SECTION
# ------------------------------
st.header("📊 Predict Sales")

item_weight = st.number_input("Item Weight", min_value=0.0, value=10.0)
item_visibility = st.number_input("Item Visibility", min_value=0.0, value=0.05)
item_mrp = st.number_input("Item MRP", min_value=0.0, value=150.0)
outlet_age = st.number_input("Outlet Age", min_value=0, value=10)

item_fat = st.selectbox("Item Fat Content", train["Item_Fat_Content"].unique())
item_type = st.selectbox("Item Type", train["Item_Type"].unique())
outlet_size = st.selectbox("Outlet Size", train["Outlet_Size"].unique())
outlet_location = st.selectbox("Outlet Location Type", train["Outlet_Location_Type"].unique())
outlet_type = st.selectbox("Outlet Type", train["Outlet_Type"].unique())

input_df = pd.DataFrame({
    "Item_Weight": [item_weight],
    "Item_Visibility": [item_visibility],
    "Item_MRP": [item_mrp],
    "Outlet_Age": [outlet_age],
    "Item_Fat_Content": [item_fat],
    "Item_Type": [item_type],
    "Outlet_Size": [outlet_size],
    "Outlet_Location_Type": [outlet_location],
    "Outlet_Type": [outlet_type]
})

if st.button("Predict Sales"):
    prediction = pipeline.predict(input_df)
    st.success(f"Predicted Sales: ₹ {prediction[0]:,.2f}")

# ------------------------------
# BUSINESS INSIGHTS SECTION
# ------------------------------
st.header("📈 Business Insights")

# Predict full dataset
X_full = train.drop(
    ["Item_Outlet_Sales", "Item_Identifier", "Outlet_Identifier"],
    axis=1
)
train["Predicted_Sales"] = pipeline.predict(X_full)

# ------------------------------
# HEATMAP
# ------------------------------
st.subheader("🔥 Demand Heatmap (Item vs Outlet Type)")

heatmap_data = train.pivot_table(
    values="Predicted_Sales",
    index="Item_Type",
    columns="Outlet_Type",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax)
st.pyplot(fig)

# ------------------------------
# MULTI-LEVEL SEGMENTATION
# ------------------------------
st.subheader("📦 Smart Inventory Recommendation")

segment_option = st.selectbox(
    "Segment Inventory By",
    ["Outlet Type", "Outlet Location Type (Tier)", "Specific Outlet ID"]
)

if segment_option == "Outlet Type":
    segment_column = "Outlet_Type"
elif segment_option == "Outlet Location Type (Tier)":
    segment_column = "Outlet_Location_Type"
else:
    segment_column = "Outlet_Identifier"

segment_value = st.selectbox(
    f"Select {segment_column}",
    train[segment_column].unique()
)

store_data = train[train[segment_column] == segment_value].copy()

# Percentile thresholds
high_threshold = store_data["Predicted_Sales"].quantile(0.75)
low_threshold = store_data["Predicted_Sales"].quantile(0.25)

def stock_decision(x):
    if x >= high_threshold:
        return "Increase Stock"
    elif x <= low_threshold:
        return "Reduce Stock"
    else:
        return "Maintain Stock"

store_data["Stock_Action"] = store_data["Predicted_Sales"].apply(stock_decision)

# HIGH DEMAND
st.write("🔝 High Demand Products (Top 25%)")
high_demand = store_data[store_data["Stock_Action"] == "Increase Stock"] \
    .sort_values("Predicted_Sales", ascending=False)
st.dataframe(high_demand[["Item_Type", "Predicted_Sales"]])

# MEDIUM DEMAND
st.write("⚖ Moderate Demand Products (Middle 50%)")
medium_demand = store_data[store_data["Stock_Action"] == "Maintain Stock"] \
    .sort_values("Predicted_Sales", ascending=False)
st.dataframe(medium_demand[["Item_Type", "Predicted_Sales"]])

# LOW DEMAND
st.write("📉 Low Demand Products (Bottom 25%)")
low_demand = store_data[store_data["Stock_Action"] == "Reduce Stock"] \
    .sort_values("Predicted_Sales", ascending=True)
st.dataframe(low_demand[["Item_Type", "Predicted_Sales"]])

# ------------------------------
# STORE PERFORMANCE RANKING
# ------------------------------
st.subheader("🏆 Outlet Ranking by High Demand Products")

high_count = (
    train.groupby("Outlet_Type")["Predicted_Sales"]
    .apply(lambda x: (x >= x.quantile(0.75)).sum())
    .sort_values(ascending=False)
)

st.dataframe(high_count)