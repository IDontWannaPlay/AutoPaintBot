from Estimator import PoseEstimator
import time

# Initialize PoseEstimator object
estimator = PoseEstimator(camera_device_number=0)

id_detected, rvecs, tvecs = estimator.detect_markers()

if id_detected:
  for i in range(len(rvecs)):
    t = estimator.get_camera_position(rvecs[i], tvecs[i])
    print(f"Camera position: {t}")

# print(id_detected)
# print(rvecs)
# print(tvecs)