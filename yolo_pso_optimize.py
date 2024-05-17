import cv2
import numpy as np
import time
import pyswarm as ps

flag = 0
total_time_taken = 0

# Load class names
classFile = r'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model
modelConfiguration = r'yolov3.cfg'
modelWeights = r'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to find objects in the frame
def findObjects(outputs, frame, vehicle_count):
    height, width, _ = frame.shape
    if sum(vehicle_count.values()) < 147:
        detected_vehicle_count = sum(vehicle_count.values()) + 1  # Start from 1
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                class_name = classNames[classId]
                vehicle_count[class_name] += 1  # Update vehicle count for detected class
                print(f"Vehicle count: {detected_vehicle_count}, Detected: {class_name} with confidence {confidence}. Total {class_name}s: {vehicle_count[class_name]}")
                return class_name, detected_vehicle_count  # Return both class name and total count of vehicles detected
    # If no objects are detected, return None for both class name and vehicle count
    return None, None


# Parameters for object detection
n_variables = 2  # Number of variables to optimize
lb = [0.1, 0.1]  # Lower bounds for parameters confThreshold and nmsThreshold
ub = [1.0, 1.0]  # Upper bounds for parameters confThreshold and nmsThreshold

# Initialize variables
confThreshold = 0.5
nmsThreshold = 0.2
total_counters = 147
video_path = r"vehicle.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
else:
    print("Video file opened successfully.")

# Initialize vehicle count outside the objective function
vehicle_count = {class_name: 0 for class_name in classNames}

# Set the width and height of the input blob
widthHeight = 320

# Initialize variables to keep track of detected vehicles
detected_vehicles = set()

# Initialize timing variables
total_time_start = time.time()
last_vehicle_time = time.time()  # Initialize last_vehicle_time

# Process the video frame by frame
try:
    while True:  # Continue until total vehicle count reaches 147
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        output_names = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

        # Measure time taken to detect each vehicle
        vehicle_time_start = time.time()
        outputs = net.forward(output_names)

        detected_class, detected_vehicle_count = findObjects(outputs, frame, vehicle_count)

        # Calculate time taken to detect each vehicle
        vehicle_time_end = time.time()
        vehicle_time_taken = vehicle_time_end - vehicle_time_start
        print(f"Time taken to detect this vehicle: {vehicle_time_taken} seconds")

        if detected_class:
            detected_vehicles.add(detected_class)

        if sum(vehicle_count.values()) >= 147:
            break

    # Calculate total time taken to detect all vehicles
    total_time_end = time.time()
    total_time_taken = total_time_end - total_time_start
    print(f"Total time taken to detect all vehicles: {total_time_taken} seconds")
    flag = 1

except Exception as e:
    print(f"An error occurred: {e}")


# Perform PSO optimization
xopt = None
fopt = None
confThreshold = 0.5
nmsThreshold = 0.2
total_counters = 147

xopt, fopt = ps.pso(lambda x: 0.0, lb, ub, swarmsize=10, maxiter=100)  # Placeholder for PSO

# Print PSO optimization parameters
print("PSO Optimization Parameters:", xopt)