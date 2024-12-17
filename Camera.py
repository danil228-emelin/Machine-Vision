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

    green_detected = False
    green_detected2 = False

    blue_detected = False
    red_detected = False
    code = [0, 0, 0, 0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi_green is None:
            roi_green = define_roi(frame, 0.1, 0.1, 0.3, 0.3)
        if roi_blue is None:
            roi_blue = define_roi(frame, 0.5, 0.3, 0.6, 0.4)

        # Apply background subtraction to get the foreground mask
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)

        # Convert frame to HSV color space for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the HSV range for detecting green color
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Define the HSV range for detecting blue color
        lower_blue = np.array([100, 150, 50])  # Lower bound for blue color
        upper_blue = np.array([140, 255, 255])  # Upper bound for blue color

        # Define the HSV range for detecting red color
        lower_red = np.array([0, 120, 70])  # Lower red range 1
        upper_red = np.array([10, 255, 255]) # Upper red range 1

        # Create binary masks for green, blue, and red colors
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours in the green mask
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find contours in the blue mask
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find contours in the red mask
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create four equal-sized squares on the same horizontal line
        square_size = 100  # Define the size of the squares

        # First square (Blue)
        square1_top_left = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.4))  # Moved more to the left
        square1_bottom_right = (square1_top_left[0] + square_size, square1_top_left[1] + square_size)

        # Second square (Red)
        square2_top_left = (int(frame.shape[1] * 0.25), int(frame.shape[0] * 0.4))  # Moved more to the left
        square2_bottom_right = (square2_top_left[0] + square_size, square2_top_left[1] + square_size)

        # Third square (Green)
        square3_top_left = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.4))  # Moved more to the left
        square3_bottom_right = (square3_top_left[0] + square_size, square3_top_left[1] + square_size)

        # Fourth square (Purple) - New square added
        square4_top_left = (int(frame.shape[1] * 0.75), int(frame.shape[0] * 0.4))  # Placed after the third square
        square4_bottom_right = (square4_top_left[0] + square_size, square4_top_left[1] + square_size)

        # Draw the squares in red, blue, green, and purple colors
        cv2.rectangle(frame, square1_top_left, square1_bottom_right, (255, 0, 0), 2)  # First blue square
        cv2.rectangle(frame, square2_top_left, square2_bottom_right, (0, 0, 255), 2)  # Second red square
        cv2.rectangle(frame, square3_top_left, square3_bottom_right, (0, 255, 0), 2)  # Third green square
        cv2.rectangle(frame, square4_top_left, square4_bottom_right, (255, 0, 0), 2)  # Fourth blue square

        # Process green contours
        for contour in green_contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if the green object is inside the first square (ROI)
            if is_in_roi((square1_top_left, square1_bottom_right), (x, y, w, h)):
                cv2.putText(frame, "Green", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
                green_detected = True
                code[0] = 1

            if is_in_roi((square4_top_left, square4_bottom_right), (x, y, w, h)):
                cv2.putText(frame, "Green", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
                green_detected2 = True
                if (code[0] == 1 and code[1] == 1 and code[2] == 1):
                    code[3] = 1
                else:
                    code = 4 * [0]

        # Process blue contours
        for contour in blue_contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color

            # Check if the blue object is inside the second square (ROI)
            if is_in_roi((square2_top_left, square2_bottom_right), (x, y, w, h)):
                cv2.putText(frame, "Blue", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
                blue_detected = True
                if (code[0] == 1):
                    code[1] = 1
                else:
                    code = 4 * [0]

        # Process red contours inside the green square (first square)
        for contour in red_contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color

            # Check if the red object is inside the first square (ROI)
            if is_in_roi((square3_top_left, square3_bottom_right), (x, y, w, h)):
                cv2.putText(frame, "Red", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
                red_detected = True
                if (code[0] == 1 and code[1] == 1):
                    code[2] = 1
                else:
                    code = 4 * [0]

        # Check if all colors are detected
        if code[0] == 1 and code[1] == 1 and code[2] == 1 and code[3] == 1:
            cv2.putText(frame, "ALLOWED", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the foreground mask and the frame with the detected green, blue, red, and purple objects and the squares
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