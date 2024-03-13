import cv2
import math
import progressbar
from pointillism import *

class Paint:
  def __init__(self, img, res, palette, gradient, stroke_scale, grid, batch_size):
    self.img = img
    self.res = res
    self.palette = palette
    self.gradient = gradient
    self.stroke_scale = stroke_scale
    self.batch_size = batch_size
    self.grid = grid
    self.skipped = 0
    self.strokes = 0
    self.passed = np.ones_like(img) * 255

  def draw_strokes(self, reduce=False):
    bar = progressbar.ProgressBar()
    for h in bar(range(0, len(self.grid), self.batch_size)):
      pixels = np.array([self.img[x[0], x[1]] for x in self.grid[h:min(h + self.batch_size, len(self.grid))]])
      color_probabilities = compute_color_probabilities(pixels, self.palette, k=90)

      for i, (y, x) in enumerate(self.grid[h:min(h + self.batch_size, len(self.grid))]):
        self.draw_stroke(x, y, color_probabilities[i], reduce=reduce)
        if (i % 100 == 0):
          cv2.imshow("res", limit_size(self.res, 1080))
          cv2.waitKey(1)

    print("Skipped %d strokes" % self.skipped)
    print("Drew %d strokes" % self.strokes)
    cv2.imshow("res", limit_size(self.res, 1080))
    cv2.imshow("passed", limit_size(self.passed, 1080))
    cv2.waitKey(0)

  def draw_stroke(self, x, y, color_probability, reduce=False):
    color = np.around(color_select(color_probability, self.palette))
    curr = self.res[y, x]
    
    self.strokes += 1
    angle = math.degrees(self.gradient.direction(y, x)) + 90
    length = int(round(2 * self.stroke_scale + 1 * self.stroke_scale * math.sqrt(self.gradient.magnitude(y, x)) / 2))

    half_length = length // 2
    dx = half_length * math.cos(angle)
    dy = half_length * math.sin(angle)
    start_point = (int(x - dx), int(y - dy))
    end_point = (int(x + dx), int(y + dy))

    if (color.astype(np.uint8) == self.res[y, x]).all() and reduce:
      cv2.line(self.passed, start_point, end_point, color, self.stroke_scale)
      self.skipped += 1
      return
    cv2.line(self.res, start_point, end_point, color, self.stroke_scale)
    