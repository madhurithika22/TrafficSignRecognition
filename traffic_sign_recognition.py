#shell_1
pip install tensorflow opencv-python

#shell_2
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#shell_3
!unzip /content/drive/MyDrive/AI/gtsrb.zip -d /content/gtsrb_data

#shell_4
DATA_PATH = "/content/gtsrb_data/Train/"
IMG_HEIGHT, IMG_WIDTH = 32, 32  # Resizing images for consistency

# Load image data and labels
images = []
labels = []

for class_id in range(43):  # GTSRB has 43 classes
    class_path = os.path.join(DATA_PATH, str(class_id))
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        images.append(img)
        labels.append(class_id)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

#shell_5
# Normalize pixel values to [0, 1]
images = images / 255.0

# One-hot encode the labels
labels = to_categorical(labels, 43)

#shell_6
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

#shell_7
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15
)

# Fit data generator to training data
datagen.fit(X_train)

#shell_8
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(X_train[i])
    plt.title(f'Class: {np.argmax(y_train[i])}')
    plt.axis('off')
plt.tight_layout()
plt.show()

#shell_9
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#shell_10
num_classes = 43  # GTSRB has 43 traffic sign classes
input_shape = (32, 32, 3)  # Image dimensions

model = Sequential()

# Convolutional Block 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Block 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Helps prevent overfitting
model.add(Dense(num_classes, activation='softmax'))  # Output layer

#shell_11
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#shell_12
epochs = 50
batch_size = 64

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test)
)

#shell_13
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#shell_14
model.save('traffic_sign_model.h5')

#shell_15
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#shell_16
from tensorflow.keras.models import load_model

model = load_model('traffic_sign_model.h5')

#shell_17
import matplotlib.pyplot as plt
import random
import numpy as np

# Load class labels (you can define this yourself or use a .csv if available)
# Example:
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road',
    13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Display random test images with predictions
n = 5  # Number of samples
plt.figure(figsize=(10, 10))
for i in range(n):
    index = random.randint(0, len(X_test) - 1)
    image = X_test[index]
    true_label = np.argmax(y_test[index])
    prediction = model.predict(image.reshape(1, 32, 32, 3))
    predicted_label = np.argmax(prediction)

    plt.subplot(1, n, i + 1)
    plt.imshow(image)
    plt.title(f"True: {classes[true_label]}\nPred: {classes[predicted_label]}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()

#shell_18
from google.colab import files
uploaded = files.upload()

#shell_19
from tensorflow.keras.preprocessing import image
import cv2

# Load the image
img_path = 'traffic.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not load image from {img_path}. Please check the path.")
else:
    img = cv2.resize(img, (32, 32))  # Resize to match training input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Reshape for model input

#shell_20
# Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

# Print result
print(f"Predicted Class: {predicted_class}")
print(f"Traffic Sign: {classes[predicted_class]}")
print(f"Confidence: {confidence * 100:.2f}%")

#shell_21
import matplotlib.pyplot as plt

img_disp = cv2.imread(img_path)
img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

plt.imshow(img_disp)
plt.title(f"Prediction: {classes[predicted_class]}\nConfidence: {confidence * 100:.2f}%", fontsize=10)
plt.axis('off')
plt.show()
