import cv2
import cv2.aruco as aruco
import numpy as np

# Load your image
inputVideo = cv2.VideoCapture()
inputVideo.open(0) # select camera device number

# Initialize ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementMaxIterations = 100
parameters.cornerRefinementMinAccuracy = 0.01
detector = aruco.ArucoDetector(aruco_dict, parameters)
length = 7.0 # aruco tag side length, arbitrary units

# Set camera parameters (you need to calibrate your camera for accurate results)
camera_matrix = np.load('PoseEstimator/calibration_matrix/camera_matrix.npy')
dist_coeffs = np.load('PoseEstimator/calibration_matrix/dist_coeffs.npy')

while (inputVideo.grab()):
  ret, img = inputVideo.read()
  imageCopy = img
  corners, ids, rejected = detector.detectMarkers(img)
  # Detect ArUco tags
  if ids is not None:
    # Calculate camera pose for each detected tag
    # objectPoints = np.array([[-length/2, length/2, 0], [length/2, length/2, 0], [length/2, -length/2, 0], [-length/2, -length/2, 0]], dtype=np.float64)
    objectPoints = np.array([[0, length/2, length/2], [0, -length/2, length/2], [0, -length/2, -length/2], [0, length/2, -length/2]], dtype=np.float64)
    img = aruco.drawDetectedMarkers(imageCopy, corners, ids)
    # print(corners)
    # Visualize the pose (e.g., draw axis on the tag)
    for i in range(len(ids)):
      _, rvec, tvec = cv2.solvePnP(objectPoints, corners[i][0], camera_matrix, dist_coeffs)
      img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 5)
      t = np.ndarray.flatten(tvec)
      r = cv2.Rodrigues(rvec)[0]
      # print(r)
      
      tagPosition = f"Tag position: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}"
      
      # Homogeneous transformation matrix 
      T = np.zeros((4,4))
      T[0:3, 0:3] = r
      T[0:3, 3] = t.T
      T[3, 3] = 1
      print(T)
      T = np.linalg.inv(T)

      t = T[0:3, 3]

      camPosition = f"Cam position: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}"
      # cv2.putText(img, camPosition, (10, (30 + i * 60) * 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
      cv2.putText(img, tagPosition, (10, (2 * 30 + i * 60) * 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

  # Display the image with pose estimation
  cv2.imshow("ArUco Detection", img)
  key = cv2.waitKey(10)

  if (key == 27):
    break