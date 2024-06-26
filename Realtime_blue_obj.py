import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Define the range of blue color in HSV
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Setup the termination criteria, either 10 iterations or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Infinite loop to process each frame from the video stream
while True:
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for blue color
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        track_window = (x, y, w, h)

        # Set up the ROI for tracking
        roi = hsv_frame[y:y+h, x:x+w]
        roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Apply MeanShift to get the new location
        ret, track_window = cv2.meanShift(mask, track_window, term_crit)

        # Draw the tracking result on the frame
        x, y, w, h = track_window
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the original frame with tracking
    cv2.imshow('Object Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
