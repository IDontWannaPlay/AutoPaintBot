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
  

  # calculate image area
  img_area = img.shape[0] * img.shape[1]
  
  # Convert image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Convert image to binary
  blur = cv2.GaussianBlur(gray,(5,5),0)
  ret, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  # dilate/erode bw image
  # set dilation/erosion kernel
  dilation_size = 3
  dilation_shape = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))
  
  # repeat dilate/erode
  dilate_erode = bw
  for i in range(3):
    dilate_erode = cv2.dilate(cv2.erode(dilate_erode, element), element)
  # dilate_erode = cv2.erode(bw, element)
  
  # # show final result
  # cv2.imshow("dilated", dilate_erode)

  # negative of dilate/erode operation
  dilate_neg = 255 - dilate_erode

  # Find all the contours in the thresholded image
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  img_hull = img
  hull_list = []

  # iterate through all contours
  for i, c in enumerate(contours):
    # ignore if contour is too small or too large
    area = cv2.contourArea(c)
    if area < 0.000 * img_area or area > 0.5 * img_area:
      continue

    # try convex hulls of contours
    hull = cv2.convexHull(c)
    hull_list.append(hull)

  # draw convex hulls for visualization
  hulls = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  for i, c in enumerate(hull_list):
    area = cv2.contourArea(c)
    if area < 0.001 * img_area or area > 0.5 * img_area:
      continue
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(hulls, hull_list, i, color, thickness=cv2.FILLED)

  dilate_mask = cv2.cvtColor(dilate_neg, cv2.COLOR_GRAY2BGR)
  hull_dilate = cv2.bitwise_and(hulls, dilate_mask)

  # do thinning, takes white on black for input so take negative
  test = cv2.cvtColor(hull_dilate, cv2.COLOR_BGR2GRAY)
  ret, test = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY)
  thinned = cv2.ximgproc.thinning(test, cv2.ximgproc.THINNING_ZHANGSUEN)
  thinned = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
  
  blended_img = cv2.addWeighted(img, 0.9, thinned, 0.9, 0)
  cv2.imshow('blended', blended_img)

  # cv2.imshow('convex hulls', hulls)
  cv2.imshow('original', img)
  # cv2.imshow('Binarized Image', bw)
  cv2.imshow('hull_dilate intersection', hull_dilate)
  cv2.imshow('thinned', thinned)
  cv2.waitKey(0)
  cv2.destroyAllWindows()