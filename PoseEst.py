import cv2
import cv2.aruco as aruco
import numpy as np


# Load your image
inputVideo = cv2.VideoCapture()
inputVideo.open(1)

# image_path = 'test_images/test_img2.jpeg'  # Replace with your image file path
# frame = cv2.imread(image_path)

# Initialize ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
aruco_square_length = 10

# Set camera parameters (you need to calibrate your camera for accurate results)
camera_matrix = np.load('calibration_matrix/camera_matrix.npy')
dist_coeffs = np.load('calibration_matrix/dist_coeffs.npy')

while (inputVideo.grab()):
  ret, img = inputVideo.read()
  imageCopy = img
  corners, ids, rejected = detector.detectMarkers(img)

  # Detect ArUco tags
  if ids is not None:
    # Calculate camera pose for each detected tag
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_square_length, camera_matrix, dist_coeffs)
    img = aruco.drawDetectedMarkers(imageCopy, corners, ids)
    # Visualize the pose (e.g., draw axis on the tag)
    for i in range(len(ids)):
      img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec[i], tvec[i], 5)
      t = tvec[0][0]
      r = cv2.Rodrigues(rvec[0][i])[0]
      
      T = np.zeros((4,4))
      T[0:3, 0:3] = r
      T[0:3, 3] = t.T
      T[3, 3] = 1

      T = np.linalg.inv(T)
      t = T[0:3, 3]

      position = f"Position: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}"
      cv2.putText(img, position, (10, (30 + 2 * i * 20) * 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


  # Display the image with pose estimation
  cv2.imshow("ArUco Detection", img)
  key = cv2.waitKey(1)

  if (key == 27):
    break