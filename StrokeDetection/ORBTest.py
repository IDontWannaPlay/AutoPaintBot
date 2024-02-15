import numpy as np
import cv2
from matplotlib import pyplot as plt

# read images
# img = cv2.imread('simple.jpg', cv2.IMREAD_GRAYSCALE)
# before = cv2.imread('simple.jpg', cv2.IMREAD_GRAYSCALE)
# after = cv2.imread('simple.jpg', cv2.IMREAD_GRAYSCALE)

before = cv2.imread('test_images/ssim_test/6_after.jpeg')
after = cv2.imread('test_images/ssim_test/6_before.jpeg')

# convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# cv2.imshow('before', before_gray)
# cv2.imshow('after', after_gray)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and compute the descriptors with ORB
kp1, des1 = orb.detectAndCompute(before_gray, None)
kp2, des2 = orb.detectAndCompute(after_gray, None)

beforekp = cv2.drawKeypoints(before, kp1, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
afterkp = cv2.drawKeypoints(after, kp2, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('beforekp', beforekp)
# cv2.imshow('afterkp', afterkp)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Converting to list for sorting as tuples are immutable objects.
matches = list(matcher.match(des1, des2, None))

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw top matches
im_matches = cv2.drawMatches(before, kp1, after, kp2, matches, None)

cv2.imshow('test', im_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Use homography to warp image
height, width, channels = before.shape
im2_reg = cv2.warpPerspective(after, h, (width, height))

# Display results
cv2.imshow('before', before)
cv2.imshow('aligned', im2_reg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('test_images/ssim_test/6_alignedv2.jpeg', im2_reg)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)

# # extract matched keypoints
# src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# # Estimate an affine transformation (ignores perspective distortion)
# M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

# if M is not None:
#   # Apply the estimated transformation to the reference image
#   aligned_reference = cv2.warpAffine(before, M, (after.shape[1], after.shape[0]))

#   # Compute the absolute difference between the aligned reference and given images
#   diff_frame = cv2.absdiff(after, aligned_reference)

#   diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

#   # Apply thresholding to highlight the differences
#   _, thresholded = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

#   # Apply morphological operations to further enhance the differences
#   thresholded = cv2.erode(thresholded, None, iterations=2)
#   thresholded = cv2.dilate(thresholded, None, iterations=2)

#   # Find contours of the differences
#   contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#   # Draw rectangles around the detected differences
#   for contour in contours:
#     if cv2.contourArea(contour) > 1000:
#       x, y, w, h = cv2.boundingRect(contour)
#       cv2.rectangle(after, (x, y), (x + w, y + h), (0, 255, 0), 2)
#       cv2.putText(after, "Change Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#       print("Significant change detected.")

#   # Display the given image with detected changes
#   cv2.imshow("Change Detected", after)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
# else:
#   print("No transformation found; images are too dissimilar.")



# # draw only keypoints location,not size and orientation
# img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()