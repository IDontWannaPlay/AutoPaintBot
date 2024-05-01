import numpy as np
import cv2
import random as rng
from skimage.metrics import structural_similarity
import time
import matplotlib as plt
from scipy.interpolate import PchipInterpolator


def align_images(img1, img2, show=False):
  """Warps img2 using homography transform to match the perspective of 
  img1. Homography transform determined with ORB.
  
    Parameters:
      img1 (numpy.ndarray): reference image perspective
      img2 (numpy.ndarray): image to match perspective of img1

    Returns:
      img2_reg (numpy.ndarray): warped version of img2 to match img1
  """
  # Convert images to grayscale
  img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  # Initiate ORB detector
  orb = cv2.ORB_create()

  # find the keypoints and compute the descriptors with ORB
  kp1, des1 = orb.detectAndCompute(img1_gray, None)
  kp2, des2 = orb.detectAndCompute(img2_gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(
    cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
  )
  
  # Converting to list for sorting as tuples are immutable objects.
  matches = list(matcher.match(des1, des2, None))
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
  
  # Remove not so good matches
  numGoodMatches = int(len(matches) * 0.1)
  matches = matches[:numGoodMatches]


  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

  # Use homography to warp image
  height, width, channels = img1.shape
  img2_reg = cv2.warpPerspective(img2, h, (width, height))

  # Display image outputs
  if (show):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv2.imshow('Image Matches', img_matches)
    # cv2.imshow('Realigned Image', img2_reg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  return img2_reg

def img_diff(img1, img2, show=False):
  """Returns a filtered SSIM image between before img1 and img2."""
  # Apply Gaussian blur on images with 5x5 kernel
  img1_blur = cv2.GaussianBlur(img1,(5,5),0)
  img2_blur = cv2.GaussianBlur(img2,(5,5),0)

  # Calculate structural similarity between images
  (score, diff) = structural_similarity(
      img1_blur, img2_blur, full=True, channel_axis=2)
  
  # diff is [0,1], scale to [0,255]
  diff = (diff * 255).astype("uint8")
  diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur on gray difference image with 5x5 kernel
  diff_blur = cv2.GaussianBlur(diff_gray,(5,5),0)
  cv2.imshow('difference blurred', diff_blur)

  # Apply threshold to blurred difference image
  # Can potentially use cv2.THRESH_OTSU
  threshold = 160
  ret, bw = cv2.threshold(diff_blur, threshold, 255, cv2.THRESH_BINARY)

  # Create dilation/erosion element
  dilation_size = int(0.008 * bw.shape[0])
  dilation_shape = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(
    dilation_shape, 
    (2 * dilation_size + 1, 2 * dilation_size + 1), 
    (dilation_size, dilation_size)
  )

  # Repeat a dilate/erode operation
  dilate_erode = bw
  for i in range(1):
    dilate_erode = cv2.erode(dilate_erode, element)
    dilate_erode = cv2.dilate(dilate_erode, element)

  # Get contours based on difference blobs
  contours, _ = cv2.findContours(dilate_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  hull_list = []
  img_area = img1.shape[0] * img1.shape[1]  # area of image

  for i, c in enumerate(contours):
    # Calculate the area of difference blob contours
    area = cv2.contourArea(c)

    # Ignore the very small or very large blobs
    if area < 0.001 * img_area or area > 0.2 * img_area:
      continue
    
    # Add convex hulls of blobs to list
    hull = cv2.convexHull(c)
    hull_list.append(hull)
  
  # Create a mask using the convex hulls
  mask = cv2.cvtColor(np.zeros_like(img1, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
  for i, c in enumerate(hull_list):
    cv2.drawContours(mask, hull_list, i, color=255, thickness=cv2.FILLED)
  
  # Create white background
  diff_masked = np.ones_like(mask, dtype=np.uint8) * 255

  # Keep masked portions of difference with blur
  diff_masked[mask != 0] = diff_blur[mask != 0]

  if (show):
    cv2.imshow("Diff blurred", diff_blur)
    cv2.imshow("Diff BW", bw)
    cv2.imshow("Dilate Eroded", dilate_erode)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return diff_masked

def get_path(img, show=False):
  """Returns image of predicted brush stroke path given image 
  resembling brush strokes. 
  """

  # Convert image to grayscale if not already
  if (img.ndim == 2):
    gray = img
  else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Convert image to binary
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  ret, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

  # Create dilation/erosion element
  dilation_size = int(0.008 * img.shape[0])
  print("dilation size: %d" % dilation_size)
  dilation_shape = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(
    dilation_shape, 
    (2 * dilation_size + 1, 2 * dilation_size + 1), 
    (dilation_size, dilation_size)
  )

  # Repeat a dilate/erode operation
  dilate_erode = bw
  for i in range(1):
    dilate_erode = cv2.erode(dilate_erode, element)
    dilate_erode = cv2.dilate(dilate_erode, element)

  # Take negative of resulting dilate/erode operation
  dilate_neg = 255 - dilate_erode

  # Find all the contours in the thresholded image
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  # Iterate through all contours
  hull_list = []
  img_area = img.shape[0] * img.shape[1]  # area of image
  for i, c in enumerate(contours):
    # Ignore if contour is too small or too large
    area = cv2.contourArea(c)
    if area < 0.001 * img_area or area > 0.5 * img_area:
      continue

    # Try convex hulls of contours
    hull = cv2.convexHull(c)
    hull_list.append(hull)

  # Draw convex hulls for visualization
  hulls = np.zeros_like(gray, dtype=np.uint8)
  for i, c in enumerate(hull_list):
    # Ignore if contour is too small or too large
    area = cv2.contourArea(c)
    if area < 0.001 * img_area or area > 0.5 * img_area:
      continue
    
    color = (rng.randint(150, 256), rng.randint(150, 256), rng.randint(150, 256))
    cv2.drawContours(hulls, hull_list, i, color, thickness=cv2.FILLED)

  # Get intersection of convex hulls and dilate/erode contours
  hull_dilate_intersect = cv2.bitwise_and(hulls, dilate_neg)
  # do thinning, takes white on black for input so take negative
  # test = cv2.cvtColor(hull_dilate_intersect, cv2.COLOR_BGR2GRAY)
  ret, test = cv2.threshold(hull_dilate_intersect, 0, 255, cv2.THRESH_BINARY)
  thinned = cv2.ximgproc.thinning(test, cv2.ximgproc.THINNING_ZHANGSUEN)

  if (show):
    cv2.imshow("Thresholded", bw)
    cv2.imshow("dilate erode", dilate_erode)
    cv2.imshow("convex hulls", hulls)
    cv2.imshow("dilate intersect", hull_dilate_intersect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return thinned

def path_pipeline(img1, img2):
  """Run align_images --> img_diff --> get_path"""
  t0 = time.time()
  aligned = align_images(img1, img2, show=True)
  t1 = time.time()
  diff_masked = img_diff(img1, aligned)
  t2 = time.time()
  path = get_path(diff_masked)
  t3 = time.time()

  print(t1-t0)
  print(t2-t1)
  print(t3-t2)
  print(t3-t0)
  return aligned, path

def polyfit_contours(reference, path, show=False):
  """takes path image and returns polynomial curves"""
  contours, _ = cv2.findContours(path, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  # Specify the degree of the polynomial fit
  n = 3  # Change this value to the desired degree
  bg = np.zeros_like(reference, dtype=np.uint8)
  for i, c in enumerate(contours):
    # Extract x and y coordinates of the contour
    xx, yy = c[:, 0, 0], c[:, 0, 1]

    # Fit a polynomial to the contour
    p = np.polyfit(xx, yy, n)

    # Generate x values for the curve
    curve_x = np.linspace(min(xx), max(xx), 100)
    curve_y = np.polyval(p, curve_x)

    # Convert curve points to integer coordinates
    curve_points = np.column_stack((curve_x.astype(int), curve_y.astype(int)))

    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.polylines(reference, [curve_points], False, color)

  cv2.imshow('Image with Curve', reference)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def bezier_fit_contours(reference, path, show=False):
  contours, _ = cv2.findContours(path, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  # Specify the number of points on the Bezier curve

  bg = np.zeros_like(reference, dtype=np.uint8)

  # Iterate over each contour and fit a Bezier curve
  for contour in contours:
    # Extract x and y coordinates of the contour
    xx, yy = contour[:, 0, 0], contour[:, 0, 1]

    # Sort the contour points based on x-values
    sorted_indices = np.argsort(xx)
    xx_sorted = xx[sorted_indices]
    yy_sorted = yy[sorted_indices]

    # Perform PCHIP interpolation
    interp_func = PchipInterpolator(xx_sorted, yy_sorted)

    # Generate x values for the curve
    curve_x = np.linspace(min(xx_sorted), max(xx_sorted), 100)
    curve_y = interp_func(curve_x)

    # Convert curve points to integer coordinates
    curve_points = np.column_stack((curve_x.astype(int), curve_y.astype(int)))

    # Draw the curve on the original image
    cv2.polylines(bg, [curve_points], isClosed=False, color=(255, 0, 0), thickness=2)

  # Display the image with the curves
  cv2.imshow('Image with Curves', bg)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



# read image
# before = cv2.imread('test_images/ssim_test/SSIMBefore.jpg')
# after = cv2.imread('test_images/ssim_test/SSIMAfter.jpg')
before = cv2.imread('test_images/ssim_test/8_before.jpeg')
after = cv2.imread('test_images/ssim_test/8_after.jpeg')

# reduce resolution
before = cv2.resize(before, (0, 0), fx=0.5, fy=0.5)
after = cv2.resize(after, (0, 0), fx=0.5, fy=0.5)

# run pipeline
aligned, path = path_pipeline(before, after)

# Superimpose path over original image
path_color = cv2.cvtColor(path, cv2.COLOR_GRAY2BGR)

polyfit_contours(aligned, path)

# find contour of line
contours, _ = cv2.findContours(path, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(contours):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(path_color, contours, i, color, thickness=1)


# image = path

# # Find non-zero elements in the image
# ii = np.nonzero(image)

# # Get y and x coordinates
# yy, xx = ii

# # Specify the degree of the polynomial fit
# n = 3  # Change this value to the desired degree

# # Fit a polynomial to the data
# p = np.polyfit(xx, yy, n)
# print(p)

# curve_x = np.linspace(0, after.shape[1], 100)
# curve_y = np.polyval(p, curve_x)

# curve_points = np.column_stack((curve_x.astype(int), curve_y.astype(int)))
# cv2.polylines(path_color, [curve_points], False, (0, 255, 0), thickness=2)
# cv2.imshow('Image with Curve', path_color)

blended_img = cv2.addWeighted(aligned, 0.4, path_color, 1, 0)

# # polyfit test
# h, w = path.shape
# ys, xs = np.nonzero(path)
# coefs = np.polyfit(xs, ys, 2)
# xx = np.arange(0, w).astype("uint8")
# yy = h - np.polyval(coefs, xx)
# color_img = np.repeat(path[:, :, np.newaxis], 3, axis=2)
# color_img[yy, xx, 0] = 255  # 0 because pyplot is RGB
# f, ax = plt.subplots(1, 2)
# ax[0].imshow(path, cmap='gray')
# ax[0].set_title('Binary')
# ax[1].imshow(color_img)
# ax[1].set_title('Polynomial')
# plt.show()

# show path line imposed on original image
cv2.imshow('brush path blended', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()