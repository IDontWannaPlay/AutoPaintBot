from Estimator import PoseEstimator
import numpy as np
import time

np.set_printoptions(precision=2, suppress=True)
# Initialize PoseEstimator object
estimator = PoseEstimator(camera_device_number=0, aruco_length_cm=5.5)

estimator.set_camera_to_brush_offset(save_to_file=False, default=True)

# Test camera position
ids, rvecs, tvecs = estimator.detect_markers(show_frame=True)
if ids is not None:
  for i in range(len(rvecs)):
    id = ids[i][0]

    canvas_to_end_effector = estimator.get_canvas_to_end_effector_matrix(rvecs[i], tvecs[i])
 
print("Canvas to end effector matrix:")


print(canvas_to_end_effector)

# while (estimator.inputVideo.grab()):
#   estimator.convert_canvas_to_end_effector()

# ids, rvecs, tvecs = estimator.detect_markers(show_frame=True)

# test_coord = np.array([0, -14.5, 0, 1])
# print(f"Test coord: {test_coord}")

# if ids is not None:
#   for i in range(len(rvecs)):
#     id = ids[i][0]
#     t = estimator.get_camera_position(rvecs[i], tvecs[i])
#     rotation_matrix = estimator.get_canvas_to_camera_matrix(rvecs[i], tvecs[i])
#     if (id == 0): 
#       rotation_matrix_0 = rotation_matrix
#     else:
#       reference_coord = t
#     print(f"Tag {id} Camera position: {t}")

#   final_coord = rotation_matrix_0 @ test_coord
#   error = final_coord[0:3] - reference_coord
#   print(f"Final coord: {final_coord[0:3]}")
#   print(f"Error: {error}")


# print(id_detected)
# print(rvecs)  
# print(tvecs)