import cv2
import cv2.aruco as aruco
import numpy as np

# Load your image
image_path = 'test_images/presentation.jpg'  # Replace with your image file path
frame = cv2.imread(image_path)

# Set camera parameters (you need to calibrate your camera for accurate results)
camera_matrix = np.load('PoseEstimator/calibration_matrix/camera_matrix_iphone.npy')
dist_coeffs = np.load('PoseEstimator/calibration_matrix/dist_coeffs_iphone.npy')

# Initialize ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
aruco_square_length = 7

# Detect ArUco tags
corners, ids, rejected = detector.detectMarkers(frame)
img = aruco.drawDetectedMarkers(frame, corners, ids)

if ids is not None:
  # Calculate camera pose for each detected tag
  length = 15
  objectPoints = np.array([[-length/2, length/2, 0], [length/2, length/2, 0], [length/2, -length/2, 0], [-length/2, -length/2, 0]], dtype=np.float64)

  # Visualize the pose (e.g., draw axis on the tag)
  for i in range(len(ids)):
    _, rvec, tvec = cv2.solvePnP(objectPoints, corners[i][0], camera_matrix, dist_coeffs)
    frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 10)
    t = np.ndarray.flatten(tvec)
    print(t)
    print(rvec)
    # position = f"Position: X={t[0]:.2f}, Y={t[1]:.2f}, Z={t[2]:.2f}"
    # cv2.putText(img, position, (10, (30 + i * 20) * 3), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)

# Display the image with pose estimation
cv2.imshow("ArUco Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
