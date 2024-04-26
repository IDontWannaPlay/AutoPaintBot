import cv2
import cv2.aruco as aruco
import numpy as np
import time

class PoseEstimator:
  def __init__(self, camera_device_number=0, aruco_side_length=7.0) -> None:
    # Load your image
    self.camera_device_number = camera_device_number

    # Initialize ArUco dictionary
    self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.01
    self.detector = aruco.ArucoDetector(self.aruco_dict, parameters)
    self.length = aruco_side_length  # aruco tag side length, arbitrary units

    # Set camera parameters (you need to calibrate camera for accurate results)
    self.camera_matrix = np.load(
        'PoseEstimator/calibration_matrix/camera_matrix.npy')
    self.dist_coeffs = np.load(
        'PoseEstimator/calibration_matrix/dist_coeffs.npy')
    
  def detect_markers(self, show_frame=False):
    # Capture a frame
    self.inputVideo = cv2.VideoCapture(self.camera_device_number)
    time.sleep(0.1)
    ret, img = self.inputVideo.read()
    self.inputVideo.release()

    # Initialize return variables
    id_detected = False
    rvecs = []  # List to store all rvec values
    tvecs = []  # List to store all tvec values
    if ret:  # if a frame has been grabbed
      image_copy = img.copy()
      corners, ids, rejected = aruco.detectMarkers(img, self.aruco_dict)
      # Detect ArUco tags
      if ids is not None:
        id_detected = True
        # Calculate camera pose for each detected tag
        objectPoints = np.array([[-self.length/2, self.length/2, 0], [self.length/2, self.length/2, 0], [self.length/2, -self.length/2, 0], [-self.length/2, -self.length/2, 0]], dtype=np.float64)
        img = aruco.drawDetectedMarkers(image_copy, corners, ids)
        # Visualize the pose (e.g., draw axis on the tag)
        for i in range(len(ids)):
          _, rvec, tvec = cv2.solvePnP(objectPoints, corners[i][0], self.camera_matrix, self.dist_coeffs)
          rvecs.append(rvec)
          tvecs.append(tvec)
          image_copy = cv2.drawFrameAxes(image_copy, self.camera_matrix, self.dist_coeffs, rvec, tvec, 5)
        
        # Display the image with pose estimation
        if show_frame:
          cv2.imshow("ArUco Detection", image_copy)
          key = cv2.waitKey(0)
          if (key == 27):
            cv2.destroyAllWindows()
      return id_detected, rvecs, tvecs
    else:
      print("Failed to grab frame.")

  def get_camera_position(self, rvec, tvec):
    t = np.ndarray.flatten(tvec)
    r = cv2.Rodrigues(rvec)[0]

    T = np.zeros((4, 4))
    T[0:3, 0:3] = r
    T[0:3, 3] = t.T
    T[3, 3] = 1

    T = np.linalg.inv(T)
    t = T[0:3, 3]
    print(T)
    return t