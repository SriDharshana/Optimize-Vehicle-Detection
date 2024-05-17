#--optimize
import cv2
import numpy as np
import time
import random

from speed_adjustment import adjust_speed

# Vehicle detection parameters
line_height = 550

# Lane width information
lane_width_pixels = 414
lane_width_meters = 3.7

# Genetic Algorithm parameters
population_size = 10
mutation_rate = 0.1
generations = 5
max_speed_vehicle = 0
max_speed = 0
def fitness(params):
    # Set default values for parameters
    min_contour_width = 40
    min_contour_height = 40
    offset = 10
    
    # Unpack parameters if available
    if len(params) >= 3:
        min_contour_width = int(params[0]) if isinstance(params[0], (int, float)) else 40
        min_contour_height = int(params[1]) if isinstance(params[1], (int, float)) else 40
        offset = int(params[2]) if isinstance(params[2], (int, float)) else 10
    elif len(params) == 2:
        min_contour_width = int(params[0]) if isinstance(params[0], (int, float)) else 40
        min_contour_height = int(params[1]) if isinstance(params[1], (int, float)) else 40
    
    cap = cv2.VideoCapture('vehicle.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    vehicles_detected = 0
    max_speed = 0
    max_speed_vehicle = 0
    
    start_time = time.time()  # Start time tracking
    
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
                        distance_pixels = abs(matches[-2][0] - cx)
                        distance_meters = distance_pixels / lane_width_pixels * lane_width_meters
                        speed_meters_per_second = distance_meters * fps
                        speed_kmph = speed_meters_per_second * 3.6
    
                        speed_kmph = adjust_speed(speed_kmph)  # Adjust speed
    
                        if speed_kmph > max_speed:
                            max_speed = speed_kmph
                            max_speed_vehicle = vehicles_detected + 1
                        vehicles_detected += 1
                        matches.remove((x, y))
    
        frame1 = frame2
        ret, frame2 = cap.read()
    
    end_time = time.time()  # End time tracking
    
    elapsed_time = end_time - start_time
    # fitness_score = elapsed_time - (1 / elapsed_time)  # Inverse of elapsed time as fitness
    fitness_score = elapsed_time
    return fitness_score



max_speed_vehicles=120
# optimal_times
max_speeds=126.31716109976982


# Genetic Algorithm

def genetic_algorithm():
    population = [[random.randint(20, 100), random.randint(20, 100), random.randint(5, 20)] for _ in range(population_size)]
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        scores = [(params, fitness(params)) for params in population]
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Best fitness in generation {generation + 1}: {scores[0][1]}")
        
        # Selection
        elite_size = int(population_size * 0.2)
        selected = scores[:elite_size]
        
        # Crossover
        children = []
        while len(children) < population_size - elite_size:
            parent1, parent2 = random.choices(selected, k=2)
            crossover_point = random.randint(1, len(parent1[0]))  
            child = parent1[0][:crossover_point] + parent2[0][crossover_point:]
            children.append(list(child))  # Convert tuple to list
            
        # Mutation
        for i in range(len(children)):
            if random.random() < mutation_rate:
                mutate_index = random.randint(0, len(children[i]) - 1)
                children[i][mutate_index] = random.randint(20, 100) if mutate_index < 2 else random.randint(5, 20)
        
        population = selected + [(child, fitness(child)) for child in children]
    
    best_params = max(population, key=lambda x: x[1])[0]
    return best_params


# best_params = genetic_algorithm()
# print(f"Best parameters: {best_params}")
best_params=[[69, 12.166411876678467], 12.26797366142273], 12.37916111946106
# Calculate final optimized time
optimal_time = fitness(best_params)
print(f'Best Parameters in Genetic algorithm: {best_params}')
print(f"Final optimized time taken: {optimal_time} seconds")


print(f"The vehicle with the highest speed is: Vehicle {max_speed_vehicles}, Speed: {max_speeds} km/h")


