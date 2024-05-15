import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Keep only car images from CIFAR-10 dataset (class label 1)
car_images = train_images[(train_labels == 1).flatten()]
car_labels = np.ones((car_images.shape[0],))

# Load and preprocess gun images
gun_images = []
gun_labels = []
gun_dir = 'gun_images'
for filename in os.listdir(gun_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = load_img(os.path.join(gun_dir, filename), target_size=(32, 32))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        gun_images.append(img_array)
        gun_labels.append(0)  # Gun class label is 0

# Ensure we have at least 4 gun images
while len(gun_images) < 4:
    if len(gun_images) > 0:
        gun_images.append(gun_images[0])  # Duplicate an existing image
        gun_labels.append(0)
    else:
        raise ValueError("Insufficient gun images")

# Convert lists to numpy arrays
gun_images = np.array(gun_images)
gun_labels = np.array(gun_labels)

# Concatenate car and gun images
X_train = np.concatenate((car_images, gun_images))
y_train = np.concatenate((car_labels, gun_labels))

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('car_gun_classifier.h5')
