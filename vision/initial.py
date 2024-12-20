import cv2
import numpy as np
from matplotlib import pyplot as plt
from .utils import *

COLOR_WHITE = (255, 255, 255)     # white
COLOR_BLACK = (0, 0, 0)           # black
COLOR_RED = (0, 0, 255)           # red
COLOR_BLUE = (255, 0, 0)          # blue
DEF_IMG_SIZE = (500, 500)         # default shape for board generation
GR_STONES_B = "BS"                  # black stones
GR_STONES_W = "WS"                  # white stones
GR_BOARD_SIZE = "BOARD_SIZE"        # board size
GR_IMG_GRAY = "IMG_GRAY"            # grayed out image
GR_IMG_BLUE = "IMG_CHANNEL_B"       # blue channel of the image
GR_IMG_RED = "IMG_CHANNEL_R"        # red channel of the image
GR_IMG_THRESH_B = "IMG_THRESH_B"    # thresholded black stones image
GR_IMG_THRESH_W = "IMG_THRESH_W"    # thresholded white stones image
GR_IMG_MORPH_B = "IMG_MORPH_B"      # morthed black stones image
GR_IMG_MORPH_W = "IMG_MORPH_W"      # thresholded white stones image
GR_IMG_LINES = "IMG_LINES1"         # generated lines image for 1st pass
GR_IMG_LINES2 = "IMG_LINES2"        # generated lines image for 2nd pass
GR_IMG_EDGES = "IMG_EDGES"          # edges image
GR_EDGES = "EDGES"                  # edges array (x,y), (x,y)
GR_SPACING = "SPACES"               # spacing of board net (x,y)
GR_NUM_LINES = "NLIN"               # overall number of lines found
GR_NUM_CROSS_H = "NCROSS_H"         # Number of crosses on horizontal line
GR_NUM_CROSS_W = "NCROSS_W"         # Number of crosses on vertical line
GR_IMAGE_SIZE = "IMAGE_SIZE"        # Image size (width, height)
GR_IMG_WS_B = "IMG_WATERSHED_B"     # Watershed black stones image
GR_IMG_WS_W = "IMG_WATERSHED_W"     # Watershed white stones image
GR_LINE_V = "LINE_V"
GR_LINE_H = "LINE_H"

def manual_warping(frame: np.array, max_dimension: int = 500):
    """
    frame : Input frame (image) to warp
    max_dimension : Maximum dimension to resize the image for easier handling (None keeps original size)
    """
    points = []  # List to store selected points

    # Mouse callback function to select points
    def select_points(event, x, y, flags, param):
        nonlocal points, image  # Access outer scope variables
        if event == cv2.EVENT_LBUTTONDOWN:
            # Append the clicked point to the list
            points.append((x, y))
            print(f"Point selected: {x}, {y}")
            # Draw a red circle at the clicked point
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  
            cv2.imshow("Select 4 Corners of the Go Board", image)

    # Resize the image for easier handling
    height, width = frame.shape[:2]
    if max_dimension is not None:
        scale_factor = min(max_dimension / width, max_dimension / height)
        new_size = (int(width * scale_factor), int(height * scale_factor))
        image = cv2.resize(frame, new_size)
    else:
        image = frame.copy()

    image_init = image.copy()  # Keep a copy of the original resized image

    # Show the resized image and set up the mouse callback to collect points
    cv2.imshow("Select 4 Corners of the Go Board", image)
    cv2.setMouseCallback("Select 4 Corners of the Go Board", select_points)

    print("Please select the 4 corners of the Go board in the order: top-left, top-right, bottom-left, bottom-right")

    # Wait until 4 points are selected
    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print("Points selected. Now warping...")

    # Source points: Selected corners of the Go board
    src_points = np.array(points, dtype=np.float32)

    # Destination points: A rectangle for the unwarped Go board
    width, height = 450, 450  # Adjust the output dimensions if needed
    dst_points = np.array([
        [0, 0],  # Top-left
        [width - 1, 0],  # Top-right
        [0, height - 1],  # Bottom-left
        [width - 1, height - 1]  # Bottom-right
    ], dtype=np.float32)

    # Compute the homography matrix
    H, _ = cv2.findHomography(src_points, dst_points)

    # Warp the image
    warped_init = cv2.warpPerspective(image_init, H, (width, height))

    return warped_init, H


def find_board(img, res, size=9):

    # Prepare gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res[GR_IMG_GRAY] = gray
    
    edges = cv2.Canny(gray, 50, 100)
    res[GR_IMG_EDGES] = edges
    # Run HoughLinesP, if its parameters are set
    # HoughLinesP detects line segments and may split a single line to multiple segments
    # The goal of running it is to remove small lines (less than minlen) which are,
    # for example, labels on board positions
    # If HoughLines is to run, its results will be used for further recognition as input
    img_detect = edges
    n_rho = 1
    n_theta = 1 * np.pi / 180
    n_thresh = 1
    n_minlen = 1
    if n_thresh > 0 and n_minlen > 0:
        lines = cv2.HoughLinesP(edges, n_rho, n_theta, n_thresh, minLineLength = n_minlen)
        lines = houghp_to_lines(lines)
        img_detect = make_lines_img(edges.shape, lines)
        img_detect = cv2.bitwise_not(img_detect)
        res[GR_IMG_LINES] = img_detect
    # Detect lines with HoughLines
    n_rho = 1
    n_theta = 1 * np.pi / 180
    n_thresh = 1
    if n_thresh < 10: n_thresh = 90       #w/a for backward compt

    # HoughLines doesn't determine coordinates, but only direction (theta) and
    # distance from (0,0) point (rho)
    lines = cv2.HoughLines(img_detect, n_rho, n_theta, n_thresh)
    lines = sorted(lines, key = lambda f: f[0][0])

    # Find vertical/horizontal lines
    lines_v = [e for e in lines if e[0][1] == 0.0 if e[0][0] > 1]
    p = round(np.pi/2 * 100, 0)
    lines_h = [e for e in lines if round(e[0][1]*100,0) == p if e[0][0] > 1]

    # Remove duplicates (lines too close to each other)
    unique_v = unique_lines(lines_v)
    unique_h = unique_lines(lines_h)

    # Convert from (rho, theta) to coordinate-based lines
    lines_v = hough_to_lines(unique_v, img.shape)
    lines_h = hough_to_lines(unique_h, img.shape)
    vcross = len(lines_v)
    hcross = len(lines_h)
    res[GR_NUM_CROSS_H] = hcross
    res[GR_NUM_CROSS_W] = vcross
    res[GR_LINE_H] = np.sort(lines_h, axis=0)
    res[GR_LINE_V] = np.sort(lines_v, axis=1)

    # Detect edges
    if len(lines_h) == 0:

       return None, None
    if len(lines_v) == 0:

       return None, None

    top_left = [int(lines_v[0][0][0]), int(lines_h[0][0][1])]
    bottom_right = [int(lines_v[-1][0][0])+1, int(lines_h[-1][0][1])+1]
    edges = [top_left, bottom_right]
    res[GR_EDGES] = edges
    # Draw a lines grid over gray image for debugging
    line_img = img1_to_img3(gray)
    line_img = make_lines_img(gray.shape, lines_v, width = 2, color = COLOR_RED, img = line_img)
    line_img = make_lines_img(gray.shape, lines_h, width = 2, color = COLOR_RED, img = line_img)
    res[GR_IMG_LINES2] = line_img
    
    space_x, space_y = board_spacing(edges, size)
    if space_x == 0 or space_y == 0:
       print("Cannot determine spacing, check params")
       return None, None

    spacing = [space_x, space_y]
    res[GR_SPACING] = spacing
    print("Detected spacing: {}".format(spacing))


    return spacing, edges


if __name__ == "__main__":
    # frame = cv2.imread("./asset/board_init.jpg")
    # warped, H = manual_warping(frame)
    # #print(H)
    # #plt.imshow(warped)
    # frame2 = cv2.imread("./asset/board_3.jpg")
    
    # frame2 = warp_planar(frame2, H, size=(450,450))
    # res=dict()
    # edges = find_board(frame2, res)
    # plt.imshow(res[GR_IMG_LINES2])
    # plt.show()
    # print(res)

    H = None
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
    

    print("Press 'c' to capture a frame, or 'q' to quit.")
    frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        if H is None:
            cv2.imshow("Ready State", frame)
        else:
            cv2.imshow("While Game", warp_planar(frame, H, (450,450)))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("quit game")
            break
        
        if key == ord('c'):  # 카메라 조정 후, 시작해서 homography 진행
            cv2.waitKey(0)
            print("ploting 4 point")
            warped_image, H = manual_warping(frame)
            res = dict()
            edges = find_board(warped_image, res, size=9)
            cv2.imshow("line detection", res[GR_IMG_LINES2])
            line_window = True

        elif key == ord('n'):
            print("do next turn")
            if line_window is True:
                cv2.destroyWindow('line detection')
            frame_curr = frame.copy()
            board = np.zeros((9,9))
            grid_points = get_grid_points(edges[0], edges[1])
            board = update_stone(frame_prev, frame_curr, board, grid_points, player=1)
            print(board)
        frame_prev = frame.copy()
    cap.release()
    cv2.destroyAllWindows()
