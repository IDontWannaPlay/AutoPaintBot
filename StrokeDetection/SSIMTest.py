import cv2
import numpy as np
from skimage.metrics import structural_similarity
import random as rng
import glob

# Load images
# before = cv2.imread('test_images/ssim_test/1_before.jpeg')
# after = cv2.imread('test_images/ssim_test/1_after.jpeg')
before = cv2.imread('test_images/ssim_test/6_alignedv2.jpeg') 
after = cv2.imread('test_images/ssim_test/6_after.jpeg')

before = cv2.GaussianBlur(before,(5,5),0)
after = cv2.GaussianBlur(after,(5,5),0)

# # Convert images to grayscale
# before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
# after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# # Compute SSIM between the two images
# (score, diff) = structural_similarity(before_gray, after_gray, full=True)
# print("Image Similarity: {:.4f}%".format(score * 100))

# Compute SSIM between the two images color
(score, diff) = structural_similarity(before, after, full=True, channel_axis=2)
print("Image Similarity: {:.4f}%".format(score * 100))

# bgray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
# agray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
# diff2 = cv2.absdiff(bgray, agray)
# cv2.imshow('ssim diff', diff)
# cv2.imshow('subtraction diff', diff2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# diff is [0,1], scale to [0,255]
diff = (diff * 255).astype("uint8")
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

diff_blur = diff_gray

diff_blur = cv2.GaussianBlur(diff_gray,(5,5),0)
cv2.imshow('diff', diff_blur)

# thinned = cv2.ximgproc.thinning(diff_blur, cv2.ximgproc.THINNING_ZHANGSUEN)
# cv2.imshow('thinned', thinned)

ret, bw = cv2.threshold(diff_blur,30,255,cv2.THRESH_BINARY)# +cv2.THRESH_OTSU)
cv2.imshow('threshold diff', bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

hull_list = []

for i, c in enumerate(contours):
  area = cv2.contourArea(c)

  if area < 200 or area > 100000:
    continue

  hull = cv2.convexHull(c)
  hull_list.append(hull)

mask = np.zeros((after.shape[0], after.shape[1], 3), dtype=np.uint8)
bg = np.ones((after.shape[0], after.shape[1]), dtype=np.uint8) * 255
for i, c in enumerate(hull_list):
  cv2.drawContours(mask, hull_list, i, color=(255,255,255), thickness=cv2.FILLED)

  color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
  cv2.drawContours(after, hull_list, i, color)

  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  box = np.intp(box)
  # cv2.drawContours(before, [box], 0, (0,0,255), 2)

# diff_blur = cv2.cvtColor(diff_blur, cv2.COLOR_GRAY2BGR)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
print(mask.shape)
cropped = cv2.bitwise_and(diff_blur, diff_blur, mask=mask)
bg[mask!=0] = diff_blur[mask!=0]
cv2.imshow('Before Image', before)
# cv2.drawContours(after, [box], 0, (0,0,255), 2)
cv2.imshow('After Image', after)
cv2.imshow('cropped Image', bg)

# cv2.imwrite('test_images/brush_detect/whiteboard.jpeg', bg)

# cv2.imshow('Binarized Image', bw)
cv2.waitKey(0)
cv2.destroyAllWindows()