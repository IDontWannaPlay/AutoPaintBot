import numpy as np
import csv

import time

start_time = time.time()

scale_factor = 1
offset_x_cm = 0
offset_y_cm = -(0 + scale_factor)
offset_brush_length_cm = 0

# Step 1: Read the CSV file and extract coordinates
data = []
with open('test_actions.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Extract x, y coordinates for each point from the first 6 values in each row
        # considering only the first 6 values
        coordinates = np.array([float(val) for val in row[:6]])
        
        # Apply scaling
        coordinates[:6:2] *= scale_factor  # Scale x values
        coordinates[1:6:2] *= scale_factor  # Scale y values

        # Apply offsets
        coordinates[:6:2] += offset_x_cm  # Add offset to x values
        coordinates[1:6:2] += offset_y_cm  # Add offset to y values

        # Reshape coordinates to 3x2 matrix (3 points, 2 coordinates each)
        coordinates = coordinates.reshape(3, 2)
        data.append(coordinates)

print(len(data))
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Step 2: Define the homogeneous transformation matrix
# Replace the values below with your transformation matrix values
transform_matrix = np.array([
    [1, 0, 0, 0],  # Replace these with your actual transformation matrix
    [0, 1, 0, 0],  # Replace these with your actual transformation matrix
    [0, 0, 1, 0],  # Replace these with your actual transformation matrix
    [0, 0, 0, 1]   # Replace these with your actual transformation matrix
])

# transform_matrix = np.array([
#     [-0.43,  0.23, - 0.87, 95.83],
#     [-0.8,   0.35,  0.49, 16.67],
#     [0.41,  0.91,  0.03, 57.38],
#     [0.,    0.,    0.,    1.]
# ])

# Step 3: Transform the coordinates using matrix multiplication
transformed_data = []
for coordinates in data:
    # Append a column of zeros for z=0 to the coordinates to make them homogeneous
    homogeneous_coordinates = np.column_stack((coordinates, np.zeros(3))).T
    # Append a row of ones to represent the homogeneous component
    homogeneous_coordinates = np.vstack((homogeneous_coordinates, np.ones(3)))
    # Apply transformation
    transformed_coordinates = np.dot(transform_matrix, homogeneous_coordinates)
    # Extract only x, y, z coordinates (omit w)
    transformed_coordinates = transformed_coordinates[:3, :3].T

    # Subtract offset_brush_length_cm from x values
    transformed_coordinates[:, 0] -= offset_brush_length_cm

    transformed_data.append(transformed_coordinates)

# Step 4: Write the transformed coordinates back to a new CSV file
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in transformed_data:
        writer.writerow(row.ravel())


