import cv2
import numpy as np
import glob
import random as rng

# Load your image directory
image_path = 'test_images/brush_detect/*'  # Replace with your image directory path

# Loop through images
for img_file in glob.glob(image_path):
  # read image 
  img = cv2.imread(img_file)
  
  # Convert image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Convert image to binary
  blur = cv2.GaussianBlur(gray,(5,5),0)
  ret, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  # Find all the contours in the thresholded image
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  img_hull = img
  hull_list = []

  # iterate through all contours
  for i, c in enumerate(contours):

    # ignore if contour is too small or too large
    area = cv2.contourArea(c)
    if area < 100 or area > 100000:
      continue

    # get min area rectangle of contour
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.intp(box)
    # cv2.drawContours(img, [box], 0, (0,0,255),2)
    # cv2.imshow('Input Image', img)

    # try convex hulls of contours
    hull = cv2.convexHull(c)
    hull_list.append(hull)

    # cv2.imshow('')
  for i in range(len(hull_list)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(img, hull_list, i, color)

  cv2.imshow('img', img)
  cv2.imshow('Binarized Image', bw)
  cv2.waitKey(0)
  cv2.destroyAllWindows()