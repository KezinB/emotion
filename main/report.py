# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os

# %%
# Load your trained model
model = tf.keras.models.load_model("model_file.h5")

# Path to your test dataset
test_dir = r"C:\Users\kezin\Downloads\archive\test"  # Update this to your test image folder

# Image parameters (should match your model's input size)
IMG_SIZE = (48, 48)  # Update if different
BATCH_SIZE = 32  # Adjust batch size as needed

# Load test images using ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Ensures labels are one-hot encoded
    color_mode='grayscale',  # Force grayscale images (fix for input shape mismatch)
    shuffle=False  
)

# Get true labels
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  # Get class names

# Predict on test data
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)



