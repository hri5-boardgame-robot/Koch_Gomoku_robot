import cv2
import numpy as np
from initial import manual_warping, find_board

# 카메라 캡처를 위한 임시 코드
def capture_frame():
    """
    Capture a single frame from the camera.
    """
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    print("Press 'c' to capture a frame, or 'q' to quit.")
    frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        cv2.imshow("Live Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Capture the frame
            print("Frame captured.")
            break
        elif key == ord('q'):  # Quit
            print("Exiting without capturing.")
            frame = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

# Main function to capture and warp
def main():
    print("Starting the camera frame capture and warping process.")
    frame = capture_frame()
    if frame is not None:
        print("Starting manual warping process...")
        warped_image, _ = manual_warping(frame)
        cv2.imshow("Warped Image", warped_image)
        print("Warped image displayed. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No frame was captured. Exiting.")

if __name__ == "__main__":
    main()