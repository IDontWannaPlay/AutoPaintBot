import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


class VectorField:
    def __init__(self, fieldx, fieldy):
        self.fieldx = fieldx
        self.fieldy = fieldy

    @staticmethod
    def from_gradient(gray):
        fieldx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 15.36
        fieldy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 15.36

        return VectorField(fieldx, fieldy)

    def get_magnitude_image(self):
        res = np.sqrt(self.fieldx**2 + self.fieldy**2)
        
        return (res * 255/np.max(res)).astype(np.uint8)

    def smooth(self, radius, iterations=1):
        s = 2*radius + 1
        for _ in range(iterations):
            self.fieldx = cv2.GaussianBlur(self.fieldx, (s, s), 0)
            self.fieldy = cv2.GaussianBlur(self.fieldy, (s, s), 0)

    def direction(self, i, j):
        return math.atan2(self.fieldy[i, j], self.fieldx[i, j])
    
    def create_direction_image(self):
        # Create an empty array for the direction values
        direction_image = np.zeros_like(self.fieldx)

        # Calculate the direction at each point
        for i in range(self.fieldx.shape[0]):
            if i % 100 == 0:
                print("Row: %d" % i)
            for j in range(self.fieldx.shape[1]):
                direction_image[i, j] = self.direction(i, j)

        # Normalize the direction values to the range 0-255
        direction_image = ((direction_image + np.pi) * 255 / (2 * np.pi)).astype(np.uint8)

        return direction_image
    
    def display_direction(self):
        # Create the direction image
        direction_image = self.create_direction_image()

        # Display the image
        plt.imshow(direction_image, cmap='hot')
        plt.colorbar(label='Direction')
        plt.show()

    def magnitude(self, i, j):
        return math.hypot(self.fieldx[i, j], self.fieldy[i, j])

    def curvature(self, i, j):
        # Calculate the first derivatives
        fx = self.fieldx[i, j]
        fy = self.fieldy[i, j]

        fxx = cv2.Laplacian(self.fieldx.astype(np.float64), cv2.CV_64F)[i, j]
        fyy = cv2.Laplacian(self.fieldy.astype(np.float64), cv2.CV_64F)[i, j]

        # Calculate the second derivatives
        # fxx = cv2.Laplacian(self.fieldx, cv2.CV_64F)[i, j]
        # fyy = cv2.Laplacian(self.fieldy, cv2.CV_64F)[i, j]

        # Calculate the curvature
        epsilon = 1e-10
        curvature = abs(fx * fyy - fy * fxx) / (fx**2 + fy**2 + epsilon)**1.5

        return curvature

    def create_curvature_image(self):
        # Create an empty array for the curvature values
        curvature_image = np.zeros_like(self.fieldx)

        # Calculate the curvature at each point
        for i in range(self.fieldx.shape[0]):
            if i % 100 == 0:
                print("Row: %d" % i)
            for j in range(self.fieldx.shape[1]):
                curvature_image[i, j] = self.curvature(i, j)

        # Normalize the curvature values to the range 0-255
        curvature_image = (curvature_image * 255 /
                        np.max(curvature_image)).astype(np.uint8)

        return curvature_image

    def display_curvature(self):
        # Create the curvature image
        curvature_image = self.create_curvature_image()

        # Display the image
        plt.imshow(curvature_image, cmap='hot')
        plt.colorbar(label='Curvature')
        plt.show()