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
parser.add_argument('img_path', nargs='?', default="Pointillism/images/campanile.jpeg")

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
print(palette)

print("Extending color palette...")
palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)]) # palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

# display the color palette
cv2.imshow("palette", palette.to_image())
cv2.waitKey(200)

print("Computing gradient...")
gradient = VectorField.from_gradient(gray)

print("Smoothing gradient...")
gradient.smooth(gradient_smoothing_radius)

print("Drawing image...")
# create a "cartonized" version of the image to use as a base for the painting
ref = cv2.medianBlur(img, 11)
res = np.ones_like(img) * 255

# define a randomized base grid of locations for the brush strokes
baseGrid = randomized_grid(img.shape[0], img.shape[1], scale=30)
print(len(baseGrid))
batch_size = 10000

base = Paint(img, res, palette, gradient, 30, baseGrid, batch_size)
base.draw_strokes()

# define a coarse grid of locations for brush strokes
img_weight = cv2.subtract(ref, res)
img_weight = cv2.cvtColor(img_weight, cv2.COLOR_BGR2GRAY)
medGrid = weighted_randomized_grid(img_weight, 10, bias=0.5)
# medGrid = randomized_grid(img.shape[0], img.shape[1], scale=10)
print(len(medGrid))
coarse = Paint(img, res, palette, gradient, 10, medGrid, batch_size)
coarse.draw_strokes(reduce=True)

# define a finer grid of locations for the brush strokes
img_weight = cv2.subtract(ref, res)
img_weight = cv2.cvtColor(img_weight, cv2.COLOR_BGR2GRAY)
fineGrid = weighted_randomized_grid(img_weight, 3, bias=0.3)

# fineGrid = randomized_grid(img.shape[0], img.shape[1], scale=2)
print(len(fineGrid))

paint = Paint(img, res, palette, gradient, 4, fineGrid, batch_size)
paint.draw_strokes(reduce=True)

# cv2.imshow("res", limit_size(res, 1080))
# cv2.imwrite(res_path, res)
# cv2.waitKey(0)
