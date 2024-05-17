import cv2
import numpy as np
import time

from speed_adjustment import adjust_speed

# Vehicle detection parameters
min_contour_width = 40  
min_contour_height = 40  
offset = 10  
line_height = 550  

# Lane width information
lane_width_pixels = 414
lane_width_meters = 3.7

# Initialize video capture
cap = cv2.VideoCapture('vehicle.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables
if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False

ret, frame1 = cap.read()
ret, frame2 = cap.read()

vehicles_detected = 0
max_speed = 0
max_speed_vehicle = 0

start_time = time.time()  # Start time for calculating processing time

while ret:
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    matches = []  # Define matches here
    
    for(i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
 
        if not contour_valid:
            continue
        cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
        cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
        centroid = (x + w // 2, y + h // 2)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cx, cy = centroid
        for (x, y) in matches:
            if y < (line_height+offset) and y > (line_height-offset):
                if len(matches) > 1:
                    vehicle_start_time = time.time()  # Start time for detecting the vehicle
                    
                    distance_pixels = abs(matches[-2][0] - cx)
                    distance_meters = distance_pixels / lane_width_pixels * lane_width_meters
                    speed_meters_per_second = distance_meters * fps
                    speed_kmph = speed_meters_per_second * 3.6
                    
                    speed_kmph = adjust_speed(speed_kmph)  # Adjust speed

                    if speed_kmph > max_speed:
                        max_speed = speed_kmph
                        max_speed_vehicle = vehicles_detected + 1
                    vehicles_detected += 1
                    
                    vehicle_end_time = time.time()  # End time for detecting the vehicle
                    vehicle_processing_time = vehicle_end_time - vehicle_start_time
                    print(f"Vehicle {vehicles_detected} Detected - Speed: {speed_kmph} km/h - Processing Time: {vehicle_processing_time} seconds")
                    matches.remove((x, y))
 
    cv2.putText(frame1, "Total Vehicles Detected: " + str(vehicles_detected), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)

    cv2.imshow("Vehicle Detection", frame1)
    if cv2.waitKey(1) == 27:
        break
    frame1 = frame2
    ret, frame2 = cap.read()

end_time = time.time()  # End time for calculating processing time
processing_time = end_time - start_time

cv2.destroyAllWindows()     
cap.release()

# Output
print(f"The vehicle with the highest speed is: Vehicle {max_speed_vehicle}, Speed: {max_speed} km/h")
print(f"Total processing time: {processing_time} seconds")
