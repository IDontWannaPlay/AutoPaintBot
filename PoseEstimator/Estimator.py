import cv2
import cv2.aruco as aruco
import numpy as np
import time

class PoseEstimator:
  def __init__(self, camera_device_number=0, aruco_length_cm=7.0) -> None:

    # Brush length in cm
    self.brush_length_cm = 15

    # Load camera to brush offset
    try:
      self.camera_to_brush_offset = np.load('calibration_files/camera_to_brush_offset.npy')
    except:
      print("Camera to brush offset not found. Using default offset.")
      self.camera_to_brush_offset = np.array([0, 8, 15, 1])

    # Set camera device number
    self.camera_device_number = camera_device_number

    # Initialize ArUco dictionary
    self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.01
    self.detector = aruco.ArucoDetector(self.aruco_dict, parameters)
    self.aruco_length_cm = aruco_length_cm  # aruco tag side length, arbitrary units

    # Set camera parameters (you need to calibrate camera for accurate results)
    self.camera_matrix = np.load(
        'calibration_files/camera_matrix.npy')
    self.dist_coeffs = np.load(
        'calibration_files/dist_coeffs.npy')
    
  def detect_markers(self, show_frame=False):
    """
    Detects ArUco markers in a captured frame and estimates the camera pose for each detected marker.

    Args:
      show_frame (bool, optional): Whether to display the image with pose estimation. Defaults to False.

    Returns:
      tuple: A tuple containing the following elements:
        - ids (numpy.ndarray): Array of marker IDs.
        - rvecs (list): List of rotation vectors for each detected marker.
        - tvecs (list): List of translation vectors for each detected marker.
    """
    
    # Capture a frame
    self.inputVideo = cv2.VideoCapture(self.camera_device_number)
    time.sleep(0.1)
    ret, img = self.inputVideo.read()
    self.inputVideo.release()

    # Initialize return variables
    rvecs = []  # List to store all rvec values
    tvecs = []  # List to store all tvec values

    if ret:  # if a frame has been grabbed
      image_copy = img.copy()
      corners, ids, rejected = aruco.detectMarkers(img, self.aruco_dict)
      
      # Detect ArUco tags
      if ids is not None:
        # Calculate camera pose for each detected tag
        # objectPoints = np.array([[0, self.aruco_length_cm/2, self.aruco_length_cm/2], [0, -self.aruco_length_cm/2, self.aruco_length_cm/2], [
        #             0, -self.aruco_length_cm/2, -self.aruco_length_cm/2], [0, self.aruco_length_cm/2, -self.aruco_length_cm/2]], dtype=np.float64)
        objectPoints = np.array([[-self.aruco_length_cm/2, self.aruco_length_cm/2, 0], [self.aruco_length_cm/2, self.aruco_length_cm/2, 0], [
                                self.aruco_length_cm/2, -self.aruco_length_cm/2, 0], [-self.aruco_length_cm/2, -self.aruco_length_cm/2, 0]], dtype=np.float64)
        
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
      
      return ids, rvecs, tvecs
    else:
      print("Failed to grab frame.")

  def get_camera_position(self, rvec, tvec):
    """
    Calculates the camera position in 3D space based on the rotation vector (rvec) and translation vector (tvec).

    Args:
      rvec (numpy.ndarray): The rotation vector.
      tvec (numpy.ndarray): The translation vector.

    Returns:
      numpy.ndarray: The camera position as a 1D array.
    """
    t = np.ndarray.flatten(tvec)
    r = cv2.Rodrigues(rvec)[0]
    T = np.zeros((4, 4))
    T[0:3, 0:3] = r
    T[0:3, 3] = t.T
    T[3, 3] = 1
    return t
  
  ### Get transformation matrices
  def get_canvas_to_camera_matrix(self, rvec, tvec):
    """
    Calculates the homogeneous transformation matrix from the canvas coordinate system to the camera coordinate system.

    Args:
      rvec (numpy.ndarray): The rotation vector.
      tvec (numpy.ndarray): The translation vector.

    Returns:
      numpy.ndarray: The homogeneous transformation matrix from the canvas to the camera coordinate system.
    """
    t = np.ndarray.flatten(tvec)
    r = cv2.Rodrigues(rvec)[0]
    T = np.zeros((4, 4))
    T[0:3, 0:3] = r
    T[0:3, 3] = t.T
    T[3, 3] = 1
    return T

  def get_camera_to_brush_matrix(self):
    camera_to_brush_matrix = np.zeros((4, 4))
    camera_to_brush_matrix[0][2] = 1
    camera_to_brush_matrix[1][0] = -1
    camera_to_brush_matrix[2][1] = -1
    camera_to_brush_matrix[3, 3] = 1
    offset = -(camera_to_brush_matrix @ self.camera_to_brush_offset)[0:3]
    camera_to_brush_matrix[0:3, 3] = offset
    return camera_to_brush_matrix
  
  def get_brush_to_end_effector_matrix(self):
    brush_to_end_effector_matrix = np.zeros((4, 4))
    brush_to_end_effector_matrix[0:3, 0:3] = np.eye(3)
    brush_to_end_effector_matrix[0:3, 3] = np.array([self.brush_length_cm, 0, 0])
    brush_to_end_effector_matrix[3, 3] = 1
    return brush_to_end_effector_matrix
  
  def get_end_effector_to_world_matrix(self, quaternion, position):
    """
    Calculate the homogeneous transformation matrix for the end effector.
    
    Parameters:
        quaternion (numpy.array): Quaternion in the form [w, x, y, z].
        position (numpy.array): Position in the form [x, y, z].
    
    Returns:
        numpy.array: 4x4 homogeneous transformation matrix.
    """
    rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
    homogeneous_matrix = np.zeros((4, 4))
    homogeneous_matrix[0:3, 0:3] = rotation_matrix
    homogeneous_matrix[0:3, 3] = position
    homogeneous_matrix[3, 3] = 1
    return homogeneous_matrix
  
  def get_canvas_to_world_matrix(self, rvec, tvec, quaternion, position):
    """
    Calculate the homogeneous transformation matrix for the canvas.
    
    Parameters:
        rvec (numpy.array): Rotation vector.
        tvec (numpy.array): Translation vector.
        quaternion (numpy.array): Quaternion in the form [w, x, y, z].
        position (numpy.array): Position in the form [x, y, z].
    Returns:
        numpy.array: 4x4 homogeneous transformation matrix.
    """
    canvas_to_camera_matrix = self.get_canvas_to_camera_matrix(rvec, tvec)
    camera_to_brush_matrix = self.get_camera_to_brush_matrix()
    brush_to_end_effector_matrix = self.get_brush_to_end_effector_matrix()
    end_effector_to_world_matrix = self.get_end_effector_to_world_matrix(quaternion, position)
    canvas_to_world_matrix = end_effector_to_world_matrix @ brush_to_end_effector_matrix @ camera_to_brush_matrix @ canvas_to_camera_matrix
    return canvas_to_world_matrix

  def get_canvas_to_end_effector_matrix(self, rvec, tvec):
    """
    Calculate the homogeneous transformation matrix for the canvas.
    
    Parameters:
        rvec (numpy.array): Rotation vector.
        tvec (numpy.array): Translation vector.
        quaternion (numpy.array): Quaternion in the form [w, x, y, z].
        position (numpy.array): Position in the form [x, y, z].
    Returns:
        numpy.array: 4x4 homogeneous transformation matrix.
    """
    canvas_to_camera_matrix = self.get_canvas_to_camera_matrix(rvec, tvec)
    # canvas_to_camera_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 15], [0, 0, 0, 1]])
    camera_to_brush_matrix = self.get_camera_to_brush_matrix()
    brush_to_end_effector_matrix = self.get_brush_to_end_effector_matrix()
    
    print("Canvas to camera matrix:")
    print(canvas_to_camera_matrix)
    print("Camera to brush matrix:")
    print(camera_to_brush_matrix)
    print("Brush to end effector matrix:")
    print(brush_to_end_effector_matrix)
    canvas_to_world_matrix = brush_to_end_effector_matrix @ camera_to_brush_matrix @ canvas_to_camera_matrix
    return canvas_to_world_matrix
  
  ### Calibration helpers
  def set_camera_to_brush_offset(self, default=False, save_to_file=False):
    """
    Sets the camera to brush offset by calculating the brush tip coordinate in camera coordinates.

    This method prompts the user to enter the 3D coordinates of the brush tip in the canvas coordinate system.
    It then detects markers in the camera frame, calculates the camera position and rotation matrix for each marker,
    and determines the brush tip coordinate in camera coordinates using the rotation matrix.

    Args:
      save_to_file (bool, optional): Whether to save the camera to brush offset to a file. Defaults to False.

    Returns:
      None
    """
    rotation_matrix = None
    if default:
      brush_canvas_coordinates = np.array([0, -10, 0, 1])
    else:
      brush_canvas_coordinates = np.zeros(4)
      print("Enter 3D coordinates:")
      for i in range(3):
        brush_canvas_coordinates[i] = float(input(f"Enter brush coordinate {i + 1}: "))
      brush_canvas_coordinates[3] = 1
    print(f"Brush tip coordinate in canvas coordinates: {brush_canvas_coordinates[0:3]}")

    ids, rvecs, tvecs = self.detect_markers(show_frame=True)
    
    if ids is not None:
      for i in range(len(rvecs)):
        if ids[i][0] == 0:
          id = ids[i][0]
          t = self.get_camera_position(rvecs[i], tvecs[i])
          rotation_matrix = self.get_canvas_to_camera_matrix(rvecs[i], tvecs[i])
          print(f"Tag {id} Camera position: {t}")

      if rotation_matrix is None:
        print("Tag not found.")
        return
      brush_camera_coordinates = rotation_matrix @ brush_canvas_coordinates
      print(f"Brush tip coordinate: {brush_camera_coordinates[0:3]}")
      self.camera_to_brush_offset = brush_camera_coordinates
      if save_to_file:
        np.save('calibration_files/camera_to_brush_offset.npy', self.camera_to_brush_offset)

  def convert_canvas_to_world(self, rvec, tvec, quaternion, position, canvas_coords):
    """
    Convert canvas coordinates to world coordinates.
    
    Parameters:
        rvec (numpy.array): Rotation vector.
        tvec (numpy.array): Translation vector.
        quaternion (numpy.array): Quaternion in the form [w, x, y, z].
        position (numpy.array): Position in the form [x, y, z].
        canvas_coords (numpy.array): Canvas coordinates in the form [x, y, z].
    
    Returns:
        numpy.array: World coordinates in the form [x, y, z].
    """
    canvas_to_world_matrix = self.get_canvas_to_world_matrix(rvec, tvec, quaternion, position)
    canvas_coords = np.append(canvas_coords, 1)
    world_coords = canvas_to_world_matrix @ canvas_coords
    return world_coords[0:3]

  def convert_canvas_to_end_effector(self, rvec, tvec, canvas_coords):
    """
    Convert canvas coordinates to end effector coordinates.
    
    Parameters:
        rvec (numpy.array): Rotation vector.
        tvec (numpy.array): Translation vector.
        canvas_coords (numpy.array): Canvas coordinates in the form [x, y, z].
    
    Returns:
        numpy.array: End effector coordinates in the form [x, y, z].
    """
    canvas_to_end_effector_matrix = self.get_canvas_to_end_effector_matrix(rvec, tvec)
    canvas_coords = np.append(canvas_coords, 1)
    end_effector_coords = canvas_to_end_effector_matrix @ canvas_coords
    return end_effector_coords[0:3]

  ### some other helpers
  def quaternion_to_rotation_matrix(self, quaternion):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Parameters:
        q (numpy.array): Quaternion in the form [w, x, y, z].
    
    Returns:
        numpy.array: 3x3 rotation matrix.
    """
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix
