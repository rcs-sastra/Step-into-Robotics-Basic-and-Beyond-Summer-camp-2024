

import cv2
import numpy as np

# Read an image from file
image = cv2.imread('image.jpg')

# Display the image
cv2.imshow('Original Image', image)

#image properties
height, width, channels = image.shape #shape[0] will give height and shape[1] will give width
print(f"Image Dimensions: {width} x {height}")
print(f"Number of Channels: {channels}")
print(f"Image Data Type: {image.dtype}")

# Crop the image (parameters: y_start, y_end, x_start, x_end)
cropped_image = image[50:200, 100:300]

# Grayscale conversion
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)


# Display the original image
cv2.imshow('Original Image', image)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)

# Display the Gaussian blurred image
cv2.imshow('Gaussian Blur', gaussian_blur)

# Resize the image to half its original size
resized_image = cv2.resize(image, (width // 2 , height // 2))

# Rotate the image 90 degrees clockwise
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE )

# Display the resized image
cv2.imshow('Resized Image', resized_image)

# Display the rotated image
cv2.imshow('Rotated Image', rotated_image)

cv2.imshow('Cropped Image', cropped_image)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
