# Optimize-Vehicle-Detection
Image processing and computer vision play pivotal roles across various domains, enhancing image quality and extracting valuable insights. In the realm of traffic management, efficient handling of increasing vehicular traffic is imperative for safety and congestion mitigation. This implementation explores diverse techniques and technologies for vehicle detection and counting in varied traffic scenarios. Leveraging Python's OpenCV library, improved YOLO v5 networks and data enhancement algorithms, are implemented to achieve high accuracy in detection and counting, addressing challenges like occlusion and small target detection. Furthermore, incorporating speed calculation, detection time analysis, vehicle type counting, and lane information, essential for effective traffic management. Experimental results, derived from real-time video captured by a single camera, underscore the system's potential for practical applications in highway management and traffic control. Additionally, the integration of traffic signal systems with detection data transmission enhances traffic flow optimization. Utilizing a combination of techniques, including YOLO, Random Forest with VGG16, and optimization methods such as Particle Swarm Intelligence, Genetic Algorithm, Simulated Annealing, and Harmony Search Optimization, ensures robust performance and accuracy in traffic analysis and management.

Optimize vehicle detection using algorithm along with the optimization techniques like **genetic algorithm optimization, particle swarm optimization(PSO), harmonic optimization** and comparing these result with the algorithm like **computer vision, yolo, random forest with VGG16** without adding any optimization techniques

# PROPOSED METHODOLOGY:
•	Open CV

In this work, OpenCV is used for detecting vehicles, counting vehicles, identifying the speed of each vehicle, measuring the time taken to detect each vehicle, identifying the vehicle with the highest speed, calculating the highest speed among all vehicles, and determining the total processing time for detecting every vehicle.

•	Open CV with Genetic Algorithm Optimization

In this work, OpenCV is utilized for detecting vehicles, counting vehicles, identifying the speed of each vehicle, measuring the time taken to detect each vehicle, identifying the vehicle with the highest speed, calculating the highest speed among all vehicles, and determining the total processing time for detecting every vehicle. Here, to speed up the detection time of total vehicles, we have integrated Genetic Algorithm optimization. This algorithm focuses on optimizing the contour width, contour height, and offset parameters. This results in a reduction in the time required to detect the total number of vehicles.

•	YOLO

YOLO (You Only Look Once) is used for detecting and counting the number of vehicles. It also identifies the type of vehicle, counts the types of vehicles, calculates the total count, and determines the total processing time for detecting every vehicle.

•	YOLO with Particle Swarm Optimization

YOLO (You Only Look Once) is used for detecting and counting the number of vehicles. It also identifies the type of vehicle, counts the types of vehicles, calculates the total count, and determines the total processing time for detecting every vehicle. Here, Particle Swarm Optimization is used for optimizing the threshold values of the implementation. These threshold values include the confidence threshold value and the non-maximum suppression threshold value. This results in a reduction in the time required for detecting the total number of vehicles compared to YOLO without optimization techniques
.
•	YOLO with Simulated Annealing

YOLO (You Only Look Once) is used for detecting and counting the number of vehicles. It also identifies the type of vehicle, counts the types of vehicles, calculates the total count, and determines the total processing time for detecting every vehicle. Here, Simulated Annealing is used for optimizing parameters such as initial temperature, minimum temperature limit, and reducing rate (alpha). This results in a reduction in the time required for detecting the total number of vehicles compared to YOLO without optimization techniques.

•	Random Forest and VGG16

Here, the Stanford car dataset is used for training the model. Before training, the images are passed into the VGG16 architecture to extract features. These values are then converted into the size of VGG16 layers. Finally, these values are inputted into the Random Forest classifier model by splitting them into training and testing datasets. Then, the model is fitted, and the accuracy is predicted using the 'accuracy' metric. The final model is used for vehicle detection, where the input will be a video. This returns the total time taken for vehicle detection.

•	Random Forest and VGG16 with Harmony Search optimization 

Here, the Stanford car dataset is used for training the model. Before training, the images are passed into the VGG16 architecture to extract features. These values are then converted into the size of VGG16 layers. Finally, these values are inputted into the Random Forest classifier model by splitting them into training and testing datasets. Then, the model is fitted, and the accuracy is predicted using the 'accuracy' metric. The final model is used for vehicle detection, where the input will be a video. Harmony Search optimization is used for tuning the frames of the images so that the image will be accurately detected after obtaining the best fitness value. With those optimized values, the total time taken for vehicle detection is calculated. 


# Comparison of result

![image](https://github.com/SriDharshana/Optimize-Vehicle-Detection/assets/86719672/af46b928-1477-4e3b-85f5-c3e41b0d10a1)

**Result with only optimization technique used**

![image](https://github.com/SriDharshana/Optimize-Vehicle-Detection/assets/86719672/ca564f86-aab7-425a-8fa6-21a6923acaf5)

# YOLO result
**without optimization**
![image](https://github.com/SriDharshana/Optimize-Vehicle-Detection/assets/86719672/40c374a4-01c5-47fe-a77e-75c28112b050)

**with PSO optimization**
![image](https://github.com/SriDharshana/Optimize-Vehicle-Detection/assets/86719672/12bdf5fe-51b5-42e7-8136-d1797842f74c)
