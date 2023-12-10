import cv2
import cv2.aruco as aruco
import numpy as np
import glob

# Load your image directory
image_path = 'test_images/*'  # Replace with your image directory path

# Set camera parameters (you need to calibrate your camera for accurate results)
camera_matrix = np.load('calibration_matrix/camera_matrix_iphone.npy')
dist_coeffs = np.load('calibration_matrix/dist_coeffs_iphone.npy')

# Initialize ArUco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # set aruco dictionary
parameters =  aruco.DetectorParameters()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementMaxIterations = 100
parameters.cornerRefinementMinAccuracy = 0.01
detector = aruco.ArucoDetector(aruco_dict, parameters)  # initialize detector

# Loop through images
for img_file in glob.glob(image_path):
  # read image 
  frame = cv2.imread(img_file)

  # Detect ArUco tags
  corners, ids, rejected = detector.detectMarkers(frame)
  img = aruco.drawDetectedMarkers(frame, corners, ids)

  # if tag(s) detected
  if ids is not None:
    # Set tag corner coordinates
    length = 10
    objectPoints = np.array([[-length/2, length/2, 0], [length/2, length/2, 0], [length/2, -length/2, 0], [-length/2, -length/2, 0]], dtype=np.float64)

    # loop through detected tags
    for i in range(len(ids)):
      # Calculate pose of detected tag
      _, rvec, tvec = cv2.solvePnP(objectPoints, corners[0][i], camera_matrix, dist_coeffs)
      img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 5)
      
      # Homogeneous transform matrix
      t = np.ndarray.flatten(tvec)
      r = cv2.Rodrigues(rvec)[0]
      
      T = np.zeros((4,4))
      T[0:3, 0:3] = r
      T[0:3, 3] = t.T
      T[3, 3] = 1

      T = np.linalg.inv(T)
      t = T[0:3, 3]

      position = f"Position: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}"
      cv2.putText(img, position, (10, (30 + 2 * i * 20) * 4), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)
      print(position)

    # Display the image with pose estimation
    cv2.imshow("ArUco Detection", frame)
    key = cv2.waitKey(0)
    if (key == 27):
      break

cv2.destroyAllWindows()
