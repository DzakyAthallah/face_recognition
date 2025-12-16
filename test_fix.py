import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('face_recognition_model.h5')

# Load test image
img = cv2.imread('test_face.jpg')
img = cv2.resize(img, (100, 100))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img, verbose=0)[0]

print("="*60)
print("DIRECT MODEL TEST")
print("="*60)
print(f"Prediction shape: {prediction.shape}")
print(f"Raw scores: {prediction}")
print()

# Label sesuai urutan folder training (alfabet)
names = ["dani", "dzaky", "izza", "nanta", "samul"]

max_idx = np.argmax(prediction)
max_score = prediction[max_idx]
predicted_name = names[max_idx]

print(f"Max index: {max_idx}")
print(f"Max score: {max_score:.4f}")
print(f"Predicted: {predicted_name}")
print(f"All scores: {[f'{s:.4f}' for s in prediction]}")
