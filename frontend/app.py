
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"
REQUEST_TIMEOUT = 30  # seconds

# Categories the pipeline was actually fitted on. Anything outside these lists is
# handled by the encoder's handle_unknown='ignore' setting rather than crashing,
# but the prediction is less reliable, so the UI flags it.
TRAINED_STORE_TYPES = [
    "Supermarket Type1",
    "Supermarket Type2",
    "Departmental Store",
    "Food Mart",
]

# Page title
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

if Store_Type not in TRAINED_STORE_TYPES:
    st.warning(
        f"'{Store_Type}' does not appear in the training data. The encoder will treat it "
        "as an unknown category, so treat this prediction as indicative only."
    )

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}


def show_backend_error(response=None, exc=None):
    """Surface why the call failed instead of always blaming connectivity."""
    if exc is not None:
        st.error(f"Could not reach the prediction API at {BACKEND_URL}: {exc}")
        return
    try:
        detail = response.json()
    except ValueError:
        detail = response.text
    st.error(f"Prediction API returned HTTP {response.status_code}.")
    st.json(detail) if isinstance(detail, (dict, list)) else st.write(detail)


# Single Prediction
if st.button("Predict", type='primary'):

    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=product_data,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        show_backend_error(exc=exc)
    else:
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
        else:
            show_backend_error(response=response)

# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("Predict for Batch", type='primary'):

        uploaded_file.seek(0)  # rewind in case Streamlit already read the buffer
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:
            show_backend_error(exc=exc)
        else:
            if response.status_code == 200:
                results = response.json()

                st.success("Predictions completed successfully!")

                try:
                    if isinstance(results, list):
                        df = pd.DataFrame(results)
                    elif isinstance(results, dict):
                        # Check if all values are scalars
                        if all(not isinstance(v, (list, dict)) for v in results.values()):
                            df = pd.DataFrame(
                                {"Row": list(results.keys()),
                                 "Predicted_Sales": list(results.values())}
                            )
                        else:
                            df = pd.DataFrame(results)
                    else:
                        df = pd.DataFrame({"Result": [results]})

                    st.dataframe(df, use_container_width=True)

                except Exception as e:
                    st.error(f"Unable to display results as a table: {e}")
                    st.json(results)

            else:
                show_backend_error(response=response)
