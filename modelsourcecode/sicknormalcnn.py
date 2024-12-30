import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from PIL import Image  # Used for opening image to check corruption

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
CLASS_NAMES = ["normal", "sick"]
NUM_CLASSES = 2

# CNN model for binary classification
def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(1, activation='sigmoid')  # Binary classification (output = 1 neuron with sigmoid)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess images with corrupted image handling
def load_and_preprocess_images(image_dir):
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist!")
            continue
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            try:
                # Attempt to open the image with PIL to check for corruption
                with Image.open(image_path) as img:
                    img.verify()  # Verify the image integrity (without decoding it)
                # Now load and preprocess the image
                img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                labels.append(class_idx)  # 0 for "normal", 1 for "sick"
            except (IOError, SyntaxError) as e:
                # Handle corrupted or unreadable images and skip them
                print(f"Skipping corrupted image {image_path}: {e}")
                continue
    return np.array(images), np.array(labels)  # No need for one-hot encoding in binary classification

# Load data
train_dir = 'C:\\Users\\adnan\\Desktop\\IA\\projectmchach\\sick cat project\\train'
val_dir = 'C:\\Users\\adnan\\Desktop\\IA\\projectmchach\\sick cat project\\val'

X_train, Y_train = load_and_preprocess_images(train_dir)
X_val, Y_val = load_and_preprocess_images(val_dir)

# Shuffle the data
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

# Create and train the model
model = create_cnn_model()
model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_val, Y_val))

# Save the model
model.save('sicknormal.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_val, Y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
