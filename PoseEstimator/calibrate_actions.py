from Estimator import PoseEstimator
import numpy as np
import csv

def transform_actions(input_file, output_file, transform_matrix):

  scale_factor = 10 
  offset_x_cm = 12
  offset_y_cm = -15
  offset_brush_length_cm = 0

  # Step 1: Read the CSV file and extract coordinates
  data = []
  with open(input_file, 'r') as file:
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

  print("Rows read:", len(data))

  # Step 2: Transform the coordinates using matrix multiplication
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

  # Step 3: Write the transformed coordinates back to a new CSV file
  with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for row in transformed_data:
      writer.writerow(row.ravel())

# Set numpy print options
np.set_printoptions(precision=2, suppress=True)

# Initialize PoseEstimator object
estimator = PoseEstimator(camera_device_number=0, aruco_length_cm=5)

# Set camera to brush offset
print("Setting camera to brush offset...")
estimator.set_camera_to_brush_offset(load_from_file=True, default=True)

# Get the transformation matrix from canvas to world coordinates
print("Getting the transformation matrix from canvas to world coordinates...")
ids, rvecs, tvecs = estimator.detect_markers(show_frame=True)
if ids is not None:
  for i in range(len(rvecs)):
    id = ids[i][0]
    # print("Enter the end effector position and orientation (w, p, r) according to pendant:")
    # x = float(input("Enter x: ")) / 10
    # y = float(input("Enter y: ")) / 10
    # z = float(input("Enter z: ")) / 10
    # w = float(input("Enter w: "))
    # p = float(input("Enter p: "))
    # r = float(input("Enter r: "))

    x = 725.216/10
    y = 173.713/10
    z = 522.47/10
    w = -174.5
    p = -83.145
    r = 7.9
    canvas_to_world = estimator.get_canvas_to_world_matrix(rvecs[i], tvecs[i], x, y, z, w, p, r)
    canvas_to_end_effector = estimator.get_canvas_to_end_effector_matrix(rvecs[i], tvecs[i])

# print("Canvas to end effector matrix:")
# print(canvas_to_end_effector)
print("Canvas to world matrix:")
print(canvas_to_world)


# print(canvas_to_end_effector @ np.array([0, 0, 0, 1]))
# print(canvas_to_world @ np.array([0, 0, 0, 1]))
# print()
# print(canvas_to_end_effector @ np.array([10, 0, 0, 1]))
# print(canvas_to_world @ np.array([10, 0, 0, 1]))
# print()
# print(canvas_to_end_effector @ np.array([0, -10, 0, 1]))
# print(canvas_to_world @ np.array([0, -10, 0, 1]))



transform_actions("actions.csv", "output.csv", canvas_to_world)
# transform_actions("test_actions.csv", "output.csv", canvas_to_world)