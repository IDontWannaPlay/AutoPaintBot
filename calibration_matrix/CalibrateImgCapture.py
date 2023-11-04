import cv2
import cv2.aruco as aruco
import numpy as np


# Load your image
inputVideo = cv2.VideoCapture()
inputVideo.open(1)

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

  key = cv2.waitKey(1)
  if (key == 27):
    break
  elif (key == 32):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

    if ret:
      objpoints.append(objp)
      imgpoints.append(corners)
      img = cv2.drawChessboardCorners(img, (6, 6), corners, ret)
      cv2.imshow("Camera Calibration",img)
      cv2.waitKey(0)
    x += 1
    print(x)

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save camera matrix and distortion coefficients to a file
print(camera_matrix)
print(dist_coeffs)
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)