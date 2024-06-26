import cv2
# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Infinite loop to process each frame from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the original frame with tracking
    cv2.imshow('video display', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
