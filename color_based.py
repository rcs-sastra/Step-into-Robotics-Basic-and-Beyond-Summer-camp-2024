#Color based object detection
import cv2
import numpy as np

# Read an image from file
image = cv2.imread('bluecar.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Create a mask for blue color
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Bitwise-AND mask and original image
blue_objects = cv2.bitwise_and(image, image, mask=mask)

# Display the original image and the image with blue objects
cv2.imshow('Original Image', image)
cv2.imshow('Blue Objects', blue_objects)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
