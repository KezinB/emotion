# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from keras.preprocessing import image
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

# %%
train_data_dir = r"C:\Users\kezin\Downloads\dataset3\train"
validation_data_dir = r"C:\Users\kezin\Downloads\dataset3\test"

# %% [markdown]
# Define image data generators with augmentation

# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)


def augment_data(directory, target_count=3000):
    for class_label in os.listdir(directory):
        class_path = os.path.join(directory, class_label)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            num_files = len(images)
            
            # If the number of images is greater than 1000, randomly select 1000 images
            if num_files > target_count:
                print(f"Class {class_label} has {num_files} images. Randomly selecting {target_count} images.")
                images_to_keep = random.sample(images, target_count)
                for img_name in images:
                    if img_name not in images_to_keep:
                        os.remove(os.path.join(class_path, img_name))
            
            # If the number of images is less than 1000, augment the data
            elif num_files < target_count:
                print(f"Augmenting class {class_label} with {num_files} images to {target_count} images")
                datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    shear_range=0.3,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest')
                
                existing_images = []
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    existing_images.append(img)
                
                existing_images = np.vstack(existing_images)
                i = 0
                for batch in datagen.flow(existing_images, batch_size=1, save_to_dir=class_path, save_prefix='aug', save_format='jpg'):
                    i += 1
                    if i > target_count - num_files:
                        break

# Augment training data
augment_data(train_data_dir)

# %% [markdown]
# Create generators

# %%
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# %%
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_generator.__next__()

# %% [markdown]
# Define the model

# %%
# Define the model
model = Sequential()

# Convolutional layers with 5x5 kernels and Batch Normalization
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Dense layers with Batch Normalization
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# Compile the model with a custom learning rate
optimizer = Adam(learning_rate=0.001)  # Set learning rate to 0.001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# %%
# Calculate the number of training and testing images
num_train_imgs = sum([len(files) for r, d, files in os.walk(train_data_dir)])
num_test_imgs = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

print(num_train_imgs)
print(num_test_imgs)

# %%
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Stop after 10 epochs if no improvement
    restore_best_weights=True  # Restore the best weights
)

# Train the model
epochs = 100
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32,
    callbacks=[early_stopping]  # Add early stopping
)

# %% [markdown]
# Save the model

# %%
model.save('model_file.h5')

# %% [markdown]
# reports

# %%
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


