# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # ---------- DIALOGFLOW CONNECTION ----------
# import requests
# import json

# DIALOGFLOW_PROJECT_ID = "your-project-id"
# DIALOGFLOW_SESSION_ID = "123456"
# DIALOGFLOW_TOKEN = "your-dialogflow-bearer-token"

# def send_to_dialogflow(text):
#     url = f"https://dialogflow.googleapis.com/v2/projects/{DIALOGFLOW_PROJECT_ID}/agent/sessions/{DIALOGFLOW_SESSION_ID}:detectIntent"
#     headers = {
#         "Authorization": f"Bearer {DIALOGFLOW_TOKEN}",
#         "Content-Type": "application/json"
#     }
#     body = {
#         "queryInput": {
#             "text": {
#                 "text": text,
#                 "languageCode": "en-US"
#             }
#         }
#     }
#     response = requests.post(url, headers=headers, json=body)
#     result = response.json()
#     try:
#         return result['queryResult']['fulfillmentText']
#     except:
#         return "Sorry, I could not process that."

# # ---------- TEXT / VOICE ROUTE ----------
# @app.route('/sendToDialogflow', methods=['POST'])
# def handle_text():
#     data = request.get_json()
#     user_text = data.get('text', '')
#     response_text = send_to_dialogflow(user_text)
#     return jsonify({"response": response_text})

# # ---------- IMAGE UPLOAD / DISEASE PREDICTION ----------
# from PIL import Image
# import io

# def predict_disease(image_bytes):
#     # Dummy prediction for now
#     # Later replace with your trained ML model
#     return "Red Rot"  # Example disease

# @app.route('/predict-disease', methods=['POST'])
# def handle_image():
#     if 'image' not in request.files:
#         return jsonify({"disease": "No image uploaded"})
    
#     image_file = request.files['image']
#     image_bytes = image_file.read()
#     disease = predict_disease(image_bytes)
    
#     return jsonify({"disease": disease})

# # ---------- FERTILIZER / IRRIGATION LOGIC ----------
# # Example: You can expand this later
# @app.route('/fertilizer', methods=['POST'])
# def fertilizer_advice():
#     # input can be soil data, crop stage, etc.
#     return jsonify({"advice": "Use 150 kg Nitrogen per hectare"})

# @app.route('/irrigation', methods=['POST'])
# def irrigation_advice():
#     return jsonify({"advice": "Irrigate 25 mm water every 5 days"})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import io
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# FLASK SETUP
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# DIALOGFLOW CONFIG
# ==============================
DIALOGFLOW_PROJECT_ID = "your-project-id"
DIALOGFLOW_SESSION_ID = "123456"
DIALOGFLOW_TOKEN = "your-dialogflow-bearer-token"

def send_to_dialogflow(text):
    url = f"https://dialogflow.googleapis.com/v2/projects/{DIALOGFLOW_PROJECT_ID}/agent/sessions/{DIALOGFLOW_SESSION_ID}:detectIntent"
    headers = {
        "Authorization": f"Bearer {DIALOGFLOW_TOKEN}",
        "Content-Type": "application/json"
    }
    body = {
        "queryInput": {
            "text": {
                "text": text,
                "languageCode": "en-US"
            }
        }
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        result = response.json()
        return result['queryResult']['fulfillmentText']
    except:
        return "Dialogflow not configured. You said: " + text


# ==============================
# LOAD ADVICE CSV
# ==============================
advice_df = pd.read_csv("advice_data.csv")

def get_advice_from_csv(disease):
    row = advice_df[advice_df['Disease'] == disease]
    if not row.empty:
        fertilizer = row['Fertilizer_Advice'].values[0]
        irrigation = row['Irrigation_Advice'].values[0]
        return fertilizer, irrigation
    else:
        return "No fertilizer advice available", "No irrigation advice available"


# ==============================
# LOAD TRAINED MODEL
# ==============================
model = load_model("sugarcane_6class_model.h5")

# IMPORTANT: Order must match training folder order
class_names = [
    "BacterialBlights",
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
]


# ==============================
# DISEASE PREDICTION FUNCTION
# ==============================
def predict_disease(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))  # must match training size

    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Correct preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]

    return class_names[class_index]


# ==============================
# IMAGE → PREDICT → ADVICE
# ==============================
@app.route('/predict-and-advise', methods=['POST'])
def predict_and_advise():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    image_file = request.files['image']
    image_bytes = image_file.read()

    disease = predict_disease(image_bytes)
    fertilizer, irrigation = get_advice_from_csv(disease)

    return jsonify({
        "disease": disease,
        "fertilizer": fertilizer,
        "irrigation": irrigation
    })


# ==============================
# TEXT / VOICE → DIALOGFLOW
# ==============================
@app.route('/sendToDialogflow', methods=['POST'])
def handle_text():
    data = request.get_json()
    user_text = data.get('text', '')
    response_text = send_to_dialogflow(user_text)
    return jsonify({"response": response_text})


# ==============================
# OPTIONAL SIMPLE ENDPOINTS
# ==============================
@app.route('/fertilizer', methods=['POST'])
def fertilizer_advice():
    return jsonify({"advice": "Use 150 kg Nitrogen per hectare"})


@app.route('/irrigation', methods=['POST'])
def irrigation_advice():
    return jsonify({"advice": "Irrigate 25 mm water every 5 days"})


# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)