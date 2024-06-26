import cv2
import numpy as np

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Error loading image from {file_path}")
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, min_area=80):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours

def count_cars(image_path):
    # Load the image
    image = load_image(image_path)

    # Preprocess the image
    edges = preprocess_image(image)

    # Detect contours
    contours = detect_contours(edges)

    # Filter contours
    car_contours = filter_contours(contours)
    # Display the number of cars detected
    print("Number of cars detected:", len(car_contours))

    # Show the original image with bounding boxes
    cv2.imshow('Cars', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'traffic.jpg' with the path to your traffic image
count_cars('cars.jpg')
