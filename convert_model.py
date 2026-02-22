from tensorflow.keras.models import load_model

# Load your existing .h5 model
model = load_model("sugarcane_6class_model.h5")

# Export as TensorFlow SavedModel
model.export("sugarcane_model")

print("Model converted successfully!")