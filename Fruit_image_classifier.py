!pip install tensorflow numpy matplotlib Pillow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image # Used for creating dummy images

# Define folder path
dataset_path = "fruits/" # change if needed

# --- Fix: Create dummy dataset if it doesn't exist ---
# This ensures the 'fruits/' directory and its subfolders with dummy images exist
# so that ImageDataGenerator can find data.
if not os.path.exists(dataset_path):
    print(f"Creating dummy dataset in '{dataset_path}'...")
    os.makedirs(dataset_path)
    classes = ["apple", "banana", "orange"]
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        # Create a few dummy images in each class folder
        for i in range(5): # Create 5 dummy images per class
            dummy_image_path = os.path.join(class_dir, f"dummy_{class_name}_{i}.jpg")
            # Create a blank image (e.g., 128x128 pixels) with a distinct color
            color = 'red' if class_name == 'apple' else ('yellow' if class_name == 'banana' else 'orange')
            img = Image.new('RGB', (128, 128), color = color)
            img.save(dummy_image_path)
    print(f"Dummy dataset created in '{dataset_path}' with {len(classes)} classes and 5 images each.")
else:
    print(f"Dataset directory '{dataset_path}' already exists.")

# Also create a dummy test_fruit.jpg for the prediction part if it doesn't exist
test_image_path = "test_fruit.jpg"
if not os.path.exists(test_image_path):
    print(f"Creating dummy test image: '{test_image_path}'")
    img = Image.new('RGB', (128, 128), color = 'green') # A generic color for a test image
    img.save(test_image_path)
else:
    print(f"Test image '{test_image_path}' already exists.")


# Image Data Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 fruit classes
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

img = image.load_img("test_fruit.jpg", target_size=(128,128))
img_tensor = image.img_to_array(img) / 255.0
img_tensor = np.expand_dims(img_tensor, axis=0)

pred = model.predict(img_tensor)

# Dynamically get class names as sorted directory names
class_names = sorted(os.listdir(dataset_path))
print(f"Detected classes: {class_names}")
predicted_class_index = np.argmax(pred)
if predicted_class_index < len(class_names):
    print("Prediction:", class_names[predicted_class_index])
else:
    print(f"Prediction index {predicted_class_index} out of bounds for class names: {class_names}")


plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.show()

model.save("fruit_classifier.h5")