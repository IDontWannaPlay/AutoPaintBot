from Estimator import PoseEstimator
import numpy as np
import time

np.set_printoptions(precision=2, suppress=True)
# Initialize PoseEstimator object
estimator = PoseEstimator(camera_device_number=0, aruco_length_cm=5)

# estimator.set_camera_to_brush_offset(save_to_file=True, default=True)

# Test camera position
ids, rvecs, tvecs = estimator.detect_markers(show_frame=True)
if ids is not None:
  for i in range(len(rvecs)):
    id = ids[i][0]

    canvas_to_end_effector = estimator.get_canvas_to_end_effector_matrix(rvecs[i], tvecs[i])
 
# print("Canvas to end effector matrix:")
# print(canvas_to_end_effector)

x = 519.691 / 10
y = 0.788 / 10
z = 518.471 / 10
w = -54.910
p = 1.081
r = -88.633

end_to_world = estimator.get_end_effector_to_world_matrix(x, y, z, w, p, r)
print("End effector to world matrix:")
print(end_to_world)
print(end_to_world[:3,:3] @ np.array([1, 1, 1]))