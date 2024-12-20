import cv2
import numpy as np
from itertools import accumulate


def warp_planar(img, H, size, max_dimension=500):
    height, width = img.shape[:2]
    if max_dimension is not None:
        scale_factor = min(max_dimension / width, max_dimension / height)
        new_size = (int(width * scale_factor), int(height * scale_factor))
        img = cv2.resize(img, new_size)
    return cv2.warpPerspective(img, H, size)


def houghp_to_lines(lines):
    """ Transform HoughP results to lines array """
    ret = []
    if not lines is None:
        for i in lines:
            ret.append(((i[0][0], i[0][1]), (i[0][2], i[0][3])))
    return np.array(ret)


def hough_to_lines(lines, shape):
    """ Transform Hough results to lines array"""
    ret = []
    if not lines is None:
        for i in lines:
            rho, theta = i[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho

            x1 = int(x0 + shape[1]*(-b))
            y1 = int(y0 + shape[0]*(a))
            x2 = int(x0 - shape[1]*(-b))
            y2 = int(y0 - shape[0]*(a))

            ret.append(((x1, y1), (x2, y2)))
    return np.array(ret)


def unique_lines(a, delta=10):
    """Return lines which are far from each other by more than a given distance"""
    if a is None:
        return None

    v = accumulate(a, lambda x, y: x if abs(y[0][0] - x[0][0]) < delta else y)
    l = [i for i in v]

    if l is None or len(l) == 0:
        return None
    else:
        return np.unique(np.array(l), axis=0)


def make_lines_img(shape, lines, width=1, color=(0, 0, 0), img=None):
    if (img is None):
        img = np.full(shape, 255, dtype=np.uint8)
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][0]
        y2 = i[1][1]
        cv2.line(img, (x1, y1), (x2, y2), color, width)

    return img


def img1_to_img3(img):
    """ Convert 1-channel (BW) image to 3-channel"""
    if img is None:
        return None
    if len(img.shape) > 2:
        raise ValueError('Image is not 1-channel')

    img3 = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        img3[:, :, i] = img
    return img3


def board_spacing(edges, size):
    """Calculate board spacing"""
    space_x = (edges[1][0] - edges[0][0]) / float(size-1)
    space_y = (edges[1][1] - edges[0][1]) / float(size-1)
    return space_x, space_y


def update_board(prev: np.array, curr: np.array, model, player, offset=0):
    """
    prev : 이전 상태의 프레임
    curr : 현재 상태의 프레임(돌을 놓은 이후)
    state : 보드의 상태(보드 모델)
    """
    if prev.ndim == 3:  # Check if prev is a 3-channel image
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if curr.ndim == 3:  # Check if curr is a 3-channel image
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # prev 와 curr 은 모두 gray scale이어야 함
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold the difference to focus on significant changes
    _, diff_thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter out small noise by area
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 판정 위치 추정
            picked_point = (x + w//2, y+h+offset)
            distances = np.linalg.norm(
                model.grid_points - picked_point, axis=1)
            # 보드에서 가장 가까운 위치를 판정
            nearest_idx = np.argmin(distances)
            nearest_point = model.grid_points[nearest_idx]
            y, x = nearest_idx//9, nearest_idx % 9
            if model.board[y, x] == 0:
                print(f"예측 지점: {(x,y)}")
                model.board[y, x] = player

    return model


# 이 부분 디버깅 통해 수정해야 할 것 같음
def update_board_circle(curr, board, player):
    """
    Detect black stones on the board and update the board state.
    """
    # Convert to grayscale if needed
    if curr.ndim == 3:
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr.copy()

    # Apply histogram equalization
    curr_gray = cv2.medianBlur(curr_gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    curr_gray = clahe.apply(curr_gray)

    # Detect circles (stones) using Hough Circle Transform
    circles = cv2.HoughCircles(
        curr_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=15, maxRadius=30
    )
    circles = np.uint16(np.around(circles))
    curr_copy = curr.copy()
    for circle in circles[0, :]:
        x, y, r = circle
        cv2.circle(curr_copy, (x, y),
                   r, (0, 255, 0), 2)

    cv2.imshow("detect circle in update_board_circle", curr_copy)
    if circles is None:
        print("No stones detected in the current frame.")
        return board, None, None  # No circles detected

    # Process detected circles
    # circles = np.round(circles[0, :]).astype("int")

    # Filter out white stones (assuming white has higher BGR values)
    filtered_circles = []
    for circle in circles[0, :]:
        x, y, r = circle
        b, g, r = curr[y, x]  # BGR values at the circle's center

        # Check if it's not white (adjust threshold as needed)
        if b < 40 and g < 40 and r < 40:
            filtered_circles.append((x, y, r))
        # NOTE for debugging
        else:
            print(f"not black circle:{(b,g,r)}, {circle}")

    if not filtered_circles:
        print("No valid black stones detected in the current frame.")
        return board, None, None  # No black stones detected

    # Convert filtered circles back to numpy array
    filtered_circles = np.array(filtered_circles)

    # Match detected stones to grid points
    distances = np.linalg.norm(
        board.grid_points[:, np.newaxis, :] - filtered_circles[:, :2], axis=2)
    min_indices = np.argmin(distances, axis=0)

    # # Check if stones are close enough to a grid point
    # valid = distances[min_indices, np.arange(distances.shape[1])] < 20
    # if not valid.any():
    #     print("No valid stones detected near grid points.")
    #     return board, None, None

    # # Process valid stones
    # valid_indices = np.where(valid)[0]
    # occ = np.array([min_indices[idx] for idx in valid_indices])

    # if len(occ) == 0:  # No valid grid points
    #     print("No valid grid points found for detected stones.")
    #     return board, None, None

    # # Update board states for the first detected stone
    # grid_x, grid_y = board.grid_points[occ[0]]
    # detected_circle = (filtered_circles[valid_indices[0]][0],  # x
    #                    filtered_circles[valid_indices[0]][1],  # y
    #                    filtered_circles[valid_indices[0]][2])  # r
    # board.states[occ[0]] = player

    # return board, (grid_x, grid_y), detected_circle

    # NOTE turn back to original code
    nearest_grid_point_idx = np.argmin(distances, axis=0)  # Shape: (3,)
    y, x = nearest_grid_point_idx//9, nearest_grid_point_idx % 9

    # 원이 속하는 지점
    circles_grid = np.column_stack([y, x])

    # 현재 점유되어 있는 지점
    occ = np.where(np.array(board.states_loc) != 0)
    occ = np.vstack([occ[0], occ[1]]).T

    if occ.size != 0:
        # 점유된 곳은 제외시킴
        is_unoccupied = ~np.any(
            (circles_grid[:, None, :] == occ).all(axis=2), axis=1)
        if not is_unoccupied.any():
            print("돌이 놓이지 않았음, 로봇 턴으로 간주")
            return board, None, None

        unoccupied_y, unoccupied_x = circles_grid[is_unoccupied][0]
        assert len(circles_grid[is_unoccupied]) == 1, "원이 여러개 뽑혔음. 버그!"
        board.states_loc[unoccupied_y, unoccupied_x] = player
        unoccupied_circle = filtered_circles[is_unoccupied][0]
        # unoccupied_circle = circles[is_unoccupied][0]
    else:  # 시작 상태
        unoccupied_circle = filtered_circles[0]  # circles[0]
        unoccupied_y, unoccupied_x = circles_grid[0]
        board.states_loc[unoccupied_y, unoccupied_x] = player

    # assert unoccupied_circle.shape == (3,), "원이 여러개 뽑혔음. 버그!"

    return board, (unoccupied_y, unoccupied_x), unoccupied_circle


def get_grid_points(tl, br, size=9):
    """
    tl : 바둑판 좌상단 좌표
    br : 바둑판 우하단 좌표
    size : 바둑판 사이즈
    """
    x_grid = np.linspace(tl[0], br[0], size)
    y_grid = np.linspace(tl[1], br[1], size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    # Flatten the grid points into a list of coordinates
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return grid_points
