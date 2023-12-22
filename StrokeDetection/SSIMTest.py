import cv2
import numpy as np
from skimage.metrics import structural_similarity
import glob

# Load images
before = cv2.imread('test_images/ssim_test/before.jpeg')
after = cv2.imread('test_images/ssim_test/after.jpeg')

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
diff_box = cv2.merge([diff, diff, diff])
