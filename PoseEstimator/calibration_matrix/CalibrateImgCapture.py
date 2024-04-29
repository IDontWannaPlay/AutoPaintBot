import cv2
import cv2.aruco as aruco
import numpy as np


# Load your image
inputVideo = cv2.VideoCapture()
inputVideo.open(0)

# Calibration pattern
objp = np.zeros((6*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:6].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

x = 0

# Capture calibration images and prepare object and image points
while (inputVideo.grab()):
  ret, img = inputVideo.read()
  cv2.imshow("Camera Calibration", img)

  key = cv2.waitKey(3)
  if (key == 27): # stop if on escape key
    break
  elif (key == 32): # capture image on space key
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

    if ret: 
      objpoints.append(objp)
      imgpoints.append(corners)
      img = cv2.drawChessboardCorners(img, (6, 6), corners, ret)
      x += 1

      # Show detected chessboard
      cv2.imshow("Camera Calibration",img)
      cv2.waitKey(0)
      print("Calibration image: ", x)

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Show camera matrix and distance coefficients
print(camera_matrix)
print(dist_coeffs)

# Save camera matrix and distortion coefficients to a file
np.save('PoseEstimator/calibration_matrix/camera_matrix.npy', camera_matrix)
np.save('PoseEstimator/calibration_matrix/dist_coeffs.npy', dist_coeffs)