import cv2
import cv2.aruco as aruco

# Define the dictionary and tag size
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # You can choose different dictionaries
# tag_size = 200  # Set the size of the ArUco tag in pixels

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
tag_size = 200  # Set the size of the ArUco tag in pixels


# Create and save an ArUco tag image
tag_id = 1  # The ID of the tag you want to generate
tag_image = aruco.generateImageMarker(aruco_dict, tag_id, tag_size)
cv2.imwrite(f'DICT_6X6_250/aruco_tag_{tag_id}.png', tag_image)

# Display the generated tag
cv2.imshow('ArUco Tag', tag_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
