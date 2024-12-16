import cv2
import numpy as np

# Function to define the Region of Interest (ROI)
def define_roi(frame, per_a, per_b, per_c, per_d):
    h, w, _ = frame.shape
    top_left = (int(w * per_a), int(h * per_b))
    bottom_right = (int(w * per_c), int(h * per_d))
    return top_left, bottom_right

# Function to check if the bounding box is inside the ROI
def is_in_roi(roi, bbox):
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    (top_left, bottom_right) = roi
    return (top_left[0] <= center_x <= bottom_right[0]) and (top_left[1] <= center_y <= bottom_right[1])

def main():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    roi_green = None
    roi_blue = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi_green is None:
            roi_green = define_roi(frame, 0.1, 0.1, 0.3, 0.3)
        if roi_blue is None:
            roi_blue = define_roi(frame, 0.5, 0.3, 0.6, 0.4)

        # Unpack the coordinates for green ROI
        #top_left, bottom_right = roi_green
        #cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # Unpack the coordinates for blue ROI
        #top_left_blue, bottom_right_blue = roi_blue
        #cv2.rectangle(frame, top_left_blue, bottom_right_blue, (0, 0, 255), 2)

        # Apply background subtraction to get the foreground mask
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)

        # Convert frame to HSV color space for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for green detection
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create two equal-sized squares on the same horizontal line
        square_size = 100  # Define the size of the squares

        # Calculate the horizontal positions of the squares
        square1_top_left = (int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.5))  # First square
        square1_bottom_right = (square1_top_left[0] + square_size, square1_top_left[1] + square_size)

        square2_top_left = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))  # Second square
        square2_bottom_right = (square2_top_left[0] + square_size, square2_top_left[1] + square_size)

        # Draw the squares in red color
        cv2.rectangle(frame, square1_top_left, square1_bottom_right, (255, 0, 0), 2)  # First blue square
        cv2.rectangle(frame, square2_top_left, square2_bottom_right, (0, 0, 255), 2)  # Second red square

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # if is_in_roi(roi_green, (x, y, w, h)):
             #   cv2.putText(frame, "Green", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)

            if is_in_roi((square1_top_left,square1_bottom_right), (x, y, w, h)):
                cv2.putText(frame, "Green", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)

            if is_in_roi((square2_top_left,square2_bottom_right), (x, y, w, h)):
                cv2.putText(frame, "Blue", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)

        # Display the foreground mask and the frame with the detected green objects and the two squares
        cv2.imshow("Foreground Mask", fgmask)
        cv2.imshow("Frame", frame)

        # Wait for 20 milliseconds and check if the 'q' key is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function if this script is executed
if __name__ == "__main__":
    main()