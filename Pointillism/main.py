import cv2
import argparse
import math
import progressbar
from pointillism import *
from pointillism.paint import Paint


parser = argparse.ArgumentParser(description='...')
parser.add_argument('--palette-size', default=5, type=int, help="Number of colors of the base palette")
parser.add_argument('--stroke-scale', default=3, type=int, help="Scale of the brush strokes (0 = automatic)")
parser.add_argument('--gradient-smoothing-radius', default=5, type=int, help="Radius of the smooth filter applied to the gradient (0 = automatic)")
parser.add_argument('--limit-image-size', default=800, type=int, help="Limit the image size (0 = no limits)")
parser.add_argument('img_path', nargs='?', default="images/campanile.jpeg")

args = parser.parse_args()

res_path = args.img_path.rsplit(".", -1)[0] + "_drawing.jpg"
img = cv2.imread(args.img_path)

if args.limit_image_size > 0:
    img = limit_size(img, args.limit_image_size)

if args.stroke_scale == 0:
    stroke_scale = int(math.ceil(max(img.shape) / 1000))
    print("Automatically chosen stroke scale: %d" % stroke_scale)
else:
    stroke_scale = args.stroke_scale

if args.gradient_smoothing_radius == 0:
    gradient_smoothing_radius = int(round(max(img.shape) / 50))
    print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)
else:
    gradient_smoothing_radius = args.gradient_smoothing_radius

# convert the image to grayscale to compute the gradient
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Computing color palette...")
palette = ColorPalette.from_image(img, args.palette_size)

print("Extending color palette...")
palette = palette.extend([(-15, 50, 0)]) # palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

# # display the color palette
# cv2.imshow("palette", palette.to_image())
# cv2.waitKey(200)

print("Computing gradient...")
gradient = VectorField.from_gradient(gray)

print("Smoothing gradient...")
gradient.smooth(gradient_smoothing_radius)

# print("Displaying gradient...")
# gradient.display_direction()

print("Drawing image...")
# create a "cartonized" version of the image to use as a base for the painting
res = cv2.medianBlur(img, 11)
res = np.ones_like(res) * 255

# define a randomized grid of locations for the brush strokes
grid = randomized_grid(img.shape[0], img.shape[1], scale=30)
print(len(grid))
batch_size = 10000

base = Paint(img, res, palette, gradient, 30, 30, batch_size)
base.draw_strokes()

paint = Paint(img, res, palette, gradient, stroke_scale, 5, batch_size)
paint.draw_strokes()

cv2.imshow("res", limit_size(res, 1080))
cv2.imwrite(res_path, res)
cv2.waitKey(0)
