import numpy as np
import cv2
import glob
import random as rng
from skimage.metrics import structural_similarity


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
    cv2.imshow('Realigned Image', img2_reg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  return img2_reg

def img_diff(img1, img2):
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

  # Apply threshold to blurred difference image
  # Can potentially use cv2.THRESH_OTSU
  threshold = 30
  ret, bw = cv2.threshold(diff_blur, threshold, 255, cv2.THRESH_BINARY)

  # Get contours based on difference blobs
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


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

  return diff_masked

def get_path(img):
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
  dilation_size = 3
  dilation_shape = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(
    dilation_shape, 
    (2 * dilation_size + 1, 2 * dilation_size + 1), 
    (dilation_size, dilation_size)
  )

  # Repeat a dilate/erode operation
  dilate_erode = bw
  for i in range(3):
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
    if area < 0.000 * img_area or area > 0.5 * img_area:
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
    
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(hulls, hull_list, i, color, thickness=cv2.FILLED)

  # Get intersection of convex hulls and dilate/erode contours
  hull_dilate_intersect = cv2.bitwise_and(hulls, dilate_neg)

  # do thinning, takes white on black for input so take negative
  # test = cv2.cvtColor(hull_dilate_intersect, cv2.COLOR_BGR2GRAY)
  ret, test = cv2.threshold(hull_dilate_intersect, 0, 255, cv2.THRESH_BINARY)
  thinned = cv2.ximgproc.thinning(test, cv2.ximgproc.THINNING_ZHANGSUEN)

  return thinned


before = cv2.imread('test_images/ssim_test/6_before.jpeg')
after = cv2.imread('test_images/ssim_test/6_after.jpeg')

aligned = align_images(before, after)
diff_masked = img_diff(before, aligned)
diff_masked = cv2.cvtColor(diff_masked, cv2.COLOR_GRAY2BGR)
path = get_path(diff_masked)

# Superimpose path over original image
path_color = cv2.cvtColor(path, cv2.COLOR_GRAY2BGR)
blended_img = cv2.addWeighted(before, 0.9, path_color, 0.9, 0)

# cv2.imshow('aligned image', aligned)
# cv2.imshow('masked diff', diff_masked)
cv2.imshow('brush path', path)
cv2.imshow('brush path blended', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()