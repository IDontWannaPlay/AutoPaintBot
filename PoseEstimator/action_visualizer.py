import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np

# Read the CSV file and extract the points
points = []
with open('output.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        # Extract coordinates for each point
        point = [(float(row[i]), float(row[i+1]), float(row[i+2]))
                 for i in range(0, len(row), 3)]
        points.append(point)

# Flatten the points list to get all x, y, z coordinates
all_x = [p[0] for point in points for p in point]
all_y = [p[1] for point in points for p in point]
all_z = [p[2] for point in points for p in point]

# Calculate the range of x, y, and z coordinates
x_range = max(all_x) - min(all_x)
y_range = max(all_y) - min(all_y)
z_range = max(all_z) - min(all_z)

# Calculate the maximum range among x, y, and z
max_range = max(x_range, y_range, z_range)

# Calculate center for each axis
x_center = (max(all_x) + min(all_x)) / 2
y_center = (max(all_y) + min(all_y)) / 2
z_center = (max(all_z) + min(all_z)) / 2

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point
for point in points:
    x = [p[0] for p in point]
    y = [p[1] for p in point]
    z = [p[2] for p in point]
    ax.scatter(x, y, z)

# Set equal aspect ratio
ax.set_box_aspect([x_range, y_range, z_range])

# Set labels and show plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
