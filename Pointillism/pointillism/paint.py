import cv2
import math
import progressbar
from pointillism import *

class Paint:
  def __init__(self, img, res, palette, gradient, stroke_scale, grid_scale, batch_size):
    self.img = img
    self.res = res
    self.palette = palette
    self.gradient = gradient
    self.stroke_scale = stroke_scale
    self.grid_scale = grid_scale
    self.batch_size = batch_size
    self.grid = randomized_grid(img.shape[0], img.shape[1], scale=grid_scale)

  def draw_strokes(self):
    # display the color palette
    cv2.imshow("palette", self.palette.to_image())
    cv2.waitKey(200)

    bar = progressbar.ProgressBar()
    for h in bar(range(0, len(self.grid), self.batch_size)):
      pixels = np.array([self.img[x[0], x[1]] for x in self.grid[h:min(h + self.batch_size, len(self.grid))]])
      color_probabilities = compute_color_probabilities(pixels, self.palette, k=9)

      for i, (y, x) in enumerate(self.grid[h:min(h + self.batch_size, len(self.grid))]):
        self.draw_stroke(x, y, color_probabilities[i])
        if (i % 100 == 0):
          cv2.imshow("res", limit_size(self.res, 1080))
          cv2.waitKey(1)

  def draw_stroke(self, x, y, color_probability):
    color = color_select(color_probability, self.palette)
    angle = math.degrees(self.gradient.direction(y, x)) + 90
    length = int(round(2 * self.stroke_scale + self.stroke_scale * math.sqrt(self.gradient.magnitude(y, x)) / 2))

    half_length = length // 2
    dx = half_length * math.cos(angle)
    dy = half_length * math.sin(angle)
    start_point = (int(x - dx), int(y - dy))
    end_point = (int(x + dx), int(y + dy))
    cv2.line(self.res, start_point, end_point, color, self.stroke_scale)