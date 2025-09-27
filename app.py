import streamlit as st
import pandas as pd
import joblib

# Load trained model and columns
model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏠 House Price Prediction App")
st.write("Enter the house details in the sidebar and get an estimated price!")

# Sidebar inputs
st.sidebar.header("House Details")

# Numerical features
LotArea = st.sidebar.number_input("Lot Area (sq ft)", min_value=500, max_value=200000, value=8000)
OverallQual = st.sidebar.slider("Overall Quality (1=Very Poor, 10=Excellent)", 1, 10, 5)
OverallCond = st.sidebar.slider("Overall Condition (1=Very Poor, 10=Excellent)", 1, 10, 5)
YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
YearRemodAdd = st.sidebar.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2005)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=800)
GarageCars = st.sidebar.slider("Garage Capacity (cars)", 0, 5, 2)
GarageArea = st.sidebar.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=400)
FullBath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
HalfBath = st.sidebar.slider("Half Bathrooms", 0, 5, 1)
BedroomAbvGr = st.sidebar.slider("Bedrooms Above Ground", 0, 10, 3)
KitchenAbvGr = st.sidebar.slider("Kitchens Above Ground", 0, 3, 1)

# Categorical features
MSZoning = st.sidebar.selectbox("MS Zoning", ["RL", "RM", "FV", "RH", "C (all)"])
Neighborhood = st.sidebar.selectbox("Neighborhood", [
    "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", 
    "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", 
    "NridgHt", "IDOTRR", "MeadowV", "Edwards", "Timber", 
    "Gilbert", "StoneBr", "ClearCr", "NAmes", "SawyerW", 
    "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"
])
HouseStyle = st.sidebar.selectbox("House Style", [
    "1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", 
    "1.5Unf", "2.5Unf", "2.5Fin"
])
ExterQual = st.sidebar.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
KitchenQual = st.sidebar.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
GarageFinish = st.sidebar.selectbox("Garage Finish", ["Fin", "RFn", "Unf", "NA"])

# Put all inputs into a dictionary
input_dict = {
    "LotArea": [LotArea],
    "OverallQual": [OverallQual],
    "OverallCond": [OverallCond],
    "YearBuilt": [YearBuilt],
    "YearRemodAdd": [YearRemodAdd],
    "GrLivArea": [GrLivArea],
    "TotalBsmtSF": [TotalBsmtSF],
    "GarageCars": [GarageCars],
    "GarageArea": [GarageArea],
    "FullBath": [FullBath],
    "HalfBath": [HalfBath],
    "BedroomAbvGr": [BedroomAbvGr],
    "KitchenAbvGr": [KitchenAbvGr],
    "MSZoning": [MSZoning],
    "Neighborhood": [Neighborhood],
    "HouseStyle": [HouseStyle],
    "ExterQual": [ExterQual],
    "KitchenQual": [KitchenQual],
    "GarageFinish": [GarageFinish]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_dict)

# One-hot encode to match training
input_df = pd.get_dummies(input_df)

# Align columns with model training columns
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict price
if st.button("Predict House Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
