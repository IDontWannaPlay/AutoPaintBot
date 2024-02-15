import cv2
import numpy as np
from skimage.metrics import structural_similarity
import random as rng
import glob

# Load images
before = cv2.imread('test_images/ssim_test/1_before.jpeg')
after = cv2.imread('test_images/ssim_test/1_after.jpeg')

# # Convert images to grayscale
# before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
# after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# # Compute SSIM between the two images
# (score, diff) = structural_similarity(before_gray, after_gray, full=True)
# print("Image Similarity: {:.4f}%".format(score * 100))

# Compute SSIM between the two images color
(score, diff) = structural_similarity(before, after, full=True, channel_axis=2)
print("Image Similarity: {:.4f}%".format(score * 100))

# diff is [0,1], scale to [0,255]
diff = (diff * 255).astype("uint8")
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
diff_blur = cv2.GaussianBlur(diff_gray,(5,5),0)
cv2.imshow('diff', diff_blur)

# thinned = cv2.ximgproc.thinning(diff_blur, cv2.ximgproc.THINNING_ZHANGSUEN)
# cv2.imshow('thinned', thinned)

ret, bw = cv2.threshold(diff_blur,200,255,cv2.THRESH_BINARY) #+cv2.THRESH_OTSU
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

hull_list = []

for i, c in enumerate(contours):
  area = cv2.contourArea(c)

  if area < 100 or area > 100000:
    continue

  hull = cv2.convexHull(c)
  hull_list.append(hull)
  color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
  cv2.drawContours(after, hull_list, i, color)

  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  box = np.intp(box)
  # cv2.drawContours(before, [box], 0, (0,0,255), 2)
  # cv2.imshow('Before Image', before)
  # cv2.drawContours(after, [box], 0, (0,0,255), 2)
  cv2.imshow('After Image', after)

# cv2.imshow('Binarized Image', bw)
cv2.waitKey(0)
cv2.destroyAllWindows()