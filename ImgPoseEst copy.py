import cv2
import cv2.aruco as aruco
import numpy as np
import glob

# Load your image directory
image_path = 'test_images/test1/*.jpeg'  # Replace with your image file path

# Initialize ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
aruco_square_length = 10

# Set camera parameters (you need to calibrate your camera for accurate results)
camera_matrix = np.load('calibration_matrix/camera_matrix_iphone.npy')
dist_coeffs = np.load('calibration_matrix/dist_coeffs_iphone.npy')

for img_file in glob.glob(image_path):
  frame = cv2.imread(img_file)

  # Detect ArUco tags
  corners, ids, rejected = detector.detectMarkers(frame)
  img = aruco.drawDetectedMarkers(frame, corners, ids)

  if ids is not None:
    # Calculate camera pose for each detected tag
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_square_length, camera_matrix, dist_coeffs)

    # Visualize the pose (e.g., draw axis on the tag)
    for i in range(len(ids)):
      frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[i], tvec[i], 5)
      t = tvec[0][0]
      r = cv2.Rodrigues(rvec[0][i])[0]
      
      # Homogeneous transform matrix
      T = np.zeros((4,4))
      T[0:3, 0:3] = r
      T[0:3, 3] = t.T
      T[3, 3] = 1

      T = np.linalg.inv(T)
      t = T[0:3, 3]

      position = f"Position: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}"
      cv2.putText(img, position, (10, (30 + 2 * i * 20) * 4), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
      print(position)

    # Display the image with pose estimation
    cv2.imshow("ArUco Detection", frame)
    key = cv2.waitKey(0)
    if (key == 27):
      break

cv2.destroyAllWindows()
