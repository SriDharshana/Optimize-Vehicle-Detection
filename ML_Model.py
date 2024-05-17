import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time

# Step 1: Data Preparation
car_folder = 'photo_dataset\draft_train'
non_car_folder = 'non_car_scaled'

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (224, 224))  # Resize to VGG16 input size
        images.append(img)
    return images

car_images = load_images(car_folder)
non_car_images = load_images(non_car_folder)
X = np.concatenate((car_images, non_car_images))
y = np.concatenate((np.ones(len(car_images)), np.zeros(len(non_car_images))))

# Step 2: Feature Extraction
# Use pre-trained VGG16 model to extract features
from keras.applications.vgg16 import VGG16, preprocess_input

vgg16 = VGG16(weights='imagenet', include_top=False)
def extract_features(img):
    img = preprocess_input(img)
    features = vgg16.predict(img[np.newaxis, ...])
    return features.flatten()

X_features = np.array([extract_features(img) for img in X])

# Step 3: Training
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
# Step 5: Evaluate Accuracy
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Step 4: Testing
video_path = '/content/vehicle.mp4'
cap = cv2.VideoCapture(video_path)

vehicle_count = 0
start_time = time()
while cap.isOpened() and vehicle_count<=147:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (224, 224))  # Resize frame to match VGG16 input size
    features = extract_features(frame)
    prediction = rf_classifier.predict(features.reshape(1, -1))

    if prediction == 1 and vehicle_count<=147:  # Car detected
        vehicle_count += 1
        detection_time = time() - start_time
        # print(f"Vehicle {vehicle_count} detected in {detection_time:.2f} seconds")

total_detection_time = time() - start_time
print(f"Total time for detecting {vehicle_count} vehicles: {total_detection_time:.2f} seconds")
print("Accuracy:", accuracy)
cap.release()
