
# Import necessary libraries
import os
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model.
# The path is resolved relative to this file so the app works whether it is started
# from /app inside the container or from any other working directory.
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib"),
)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model artefact not found at {MODEL_PATH}. Copy superkart_model.joblib next "
        "to app.py before building the image, or set the MODEL_PATH environment variable."
    )
model = joblib.load(MODEL_PATH)  # fails fast at start-up rather than per request

# The exact feature names and order the pipeline was fitted on
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"


# Simple health check so the container can be probed before traffic is sent to it
@superkart_api.get('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    # Validate that every required field is present
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required field(s).",
            "missing_fields": missing,
            "expected_fields": FEATURES,
        }), 400

    # Extract relevant features from the input data, coercing the numeric ones
    sample = {}
    invalid = {}
    for feature in FEATURES:
        value = data[feature]
        if feature in NUMERIC_FEATURES:
            try:
                sample[feature] = float(value)
            except (TypeError, ValueError):
                invalid[feature] = value
        else:
            if value is None:
                invalid[feature] = value
            else:
                sample[feature] = str(value)
    if invalid:
        return jsonify({
            "error": "Invalid value(s) for field(s).",
            "invalid_fields": invalid,
        }), 400

    # Convert the extracted data into a DataFrame with columns in the fitted order
    input_data = pd.DataFrame([sample], columns=FEATURES)

    # Make a prediction using the trained model
    try:
        prediction = model.predict(input_data)
    except Exception:
        superkart_api.logger.exception("Single-record inference failed")
        return jsonify({"error": "Prediction failed."}), 500

    # float() converts numpy.float32/64 into a JSON-serializable Python float
    return jsonify({"Sales": round(float(np.ravel(prediction)[0]), 2)}), 200


# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get('file')
    if file is None:
        return jsonify({
            "error": "No file uploaded. Send a CSV as multipart/form-data under the key 'file'."
        }), 400

    # Read the file into a DataFrame
    try:
        input_data = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Could not parse the uploaded CSV: {exc}"}), 400

    if input_data.empty:
        return jsonify({"error": "The uploaded CSV contains no rows."}), 400

    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({
            "error": "Uploaded CSV is missing required column(s).",
            "missing_columns": missing,
            "expected_columns": FEATURES,
        }), 400

    # Reorder to the fitted column order and drop any extra columns
    input_data = input_data[FEATURES]

    # Make predictions for the batch data
    try:
        predictions = model.predict(input_data)
    except Exception:
        superkart_api.logger.exception("Batch inference failed")
        return jsonify({"error": "Batch prediction failed."}), 500

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {
        str(i): round(float(pred), 2) for i, pred in enumerate(np.ravel(predictions))
    }

    return jsonify(output_dict), 200


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
