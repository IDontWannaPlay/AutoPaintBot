import cv2
import numpy as np

# Capture calibration images and prepare object and image points

# Calibration pattern
objp = np.zeros((6*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:6].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

calibration_images = [
    'calibration_images/calib_img1.jpeg',
    'calibration_images/calib_img2.jpeg',
    'calibration_images/calib_img3.jpeg',
    'calibration_images/calib_img4.jpeg',
    'calibration_images/calib_img5.jpeg',
    'calibration_images/calib_img6.jpeg',
    'calibration_images/calib_img7.jpeg',
    # Add more image file paths as needed
]

# Capture calibration images and detect corners
for img_file in calibration_images:
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img, (6, 6), corners, ret)

    cv2.imshow('img',img)
    cv2.waitKey(0)

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save camera matrix and distortion coefficients to a file
print(camera_matrix)
print(dist_coeffs)
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)
