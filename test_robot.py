from omoku_bot_v2 import OmokuBot
import time
import numpy as np
import cv2
from vision.utils import warp_planar, update_board_circle, get_grid_points
from vision.initial import manual_warping, find_board
from Gomoku_AI.game import Board, Game
from Gomoku_AI.mcts_alphaZero import MCTSPlayer
from Gomoku_AI.policy_value_net import PolicyValueNet


class HumanPlayer:
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def print_board(self, board):
        size = board.width
        print("   " + "   ".join(map(str, range(size))))
        for i in range(size):
            row = ["." if (i * size + j) not in board.states else (
                "X" if board.states[i * size + j] == 1 else "O") for j in range(size)]
            print(f"{i}  " + "   ".join(row))

    def __str__(self):
        return "HumanPlayer"


class OmokuGame:
    def __init__(self, model_path, board_width=9, board_height=9, n_in_row=5):
        self.board = Board(width=board_width,
                           height=board_height, n_in_row=n_in_row)
        self.game = Game(self.board)

        self.policy_value_net = PolicyValueNet(board_width, board_height)
        self.policy_value_net.load_model(model_path)

        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)
        self.human_player = HumanPlayer()

        self.robot = OmokuBot(use_real_robot=True, device_name='/dev/ttyACM0')
        # self.robot.init_robot()

        self.H = None  # Homography matrix for vision
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다.")
        self.robot.reload()

    def initialize(self, edges):
        """
        Initialize the grid points for the board.
        Args:
            edges (tuple): (top-left corner, bottom-right corner) of the board.
        """
        top_left, bottom_right = edges
        grid_points = get_grid_points(
            top_left, bottom_right, size=self.board.width)
        self.board.grid_points = grid_points  # Assign grid points to the board
        print("Board grid points initialized.")

    def calibrate_board(self):
        print("Press 'c' to capture the board for calibration.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                continue

            cv2.imshow("Calibration Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                cv2.waitKey(0)
                print("Calibrating board...")
                warped_image, self.H = manual_warping(frame)
                res = {}
                spacing, edges = find_board(warped_image, res, size=9)
                if spacing[0] < 49 or spacing[1] < 49:
                    print("Calibration failed. Please retry.")
                    continue

                print("Calibration complete.")
                return edges
            elif key == ord('q'):
                print("Calibration aborted.")
                break

    def update_human_move(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            return None

        if self.H is None:
            print("Error: Homography matrix is not initialized.")
            return None

        # 프레임을 바둑판 영역으로 워핑
        frame_warped = warp_planar(frame, self.H, (450, 450))

        # NOTE debug

        # 돌 감지 및 보드 상태 업데이트
        next_board, update_point, circle = update_board_circle(
            curr=frame_warped, board=self.board, player=1
        )

        # 감지된 돌이 있을 경우
        if circle is not None:
            x, y, r = circle
            cv2.circle(frame_warped, (x, y), r, (0, 255, 0), 2)  # 감지된 돌 시각화
            print(f"Detected stone at pixel: ({x}, {y}), radius: {r}")

        # NOTE debugging detected circles
        cv2.imshow("detect circle", frame_warped)

        # 이 부분을 수정해야 할 것 같음.
        if update_point is not None:
            grid_y, grid_x = update_point

            # grid_x , grid_y => change order
            move = self.board.location_to_move([grid_y, grid_x])

            # 디버깅: 격자 좌표와 보드 상태 확인
            print(f"Human placed at grid: ({grid_x}, {grid_y})")
            print(
                f"Move index: {move}, Board states at move: {self.board.states.get(move, 'Not found')}")

            # 점유된 위치인지 확인 TODO error occured
            # if self.board.states.get(move, 0) != 0:
            #     print(f"Position ({grid_x}, {grid_y}) is already occupied.")
            #     return None  # 이미 점유된 위치

            # 보드 상태 업데이트
            self.board.states[move] = 1  # 인간 플레이어는 항상 1번
            print(f"Board updated: {self.board.states}")
            self.board = next_board
            return update_point
        else:
            print("No move detected.")
            return None

    def run(self):
        edges = self.calibrate_board()
        if not edges:
            return

        # Initialize the board grid points
        self.initialize(edges)

        start_player = 0
        self.board.init_board()
        self.human_player.set_player_ind(1)  # Human is player 1 (X)
        self.mcts_player.set_player_ind(2)  # AI is player 2 (O)

        current_player = start_player

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # 실시간 프레임 처리
            if self.H is not None:
                frame_warped = warp_planar(frame, self.H, (450, 450))
                cv2.imshow("Warped Go Board", frame_warped)
            else:
                cv2.imshow("Live View", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                if current_player == 0:  # Human's turn
                    update_point = self.update_human_move()
                    if update_point:
                        print(f"Human placed at: {update_point}")
                else:  # AI's turn
                    move = self.mcts_player.get_action(self.board)
                    self.board.do_move(move)

                    # AI의 플레이어 ID를 사용하여 보드 업데이트
                    self.board.states[move] = 2  # AI는 항상 2번 플레이어 (O)

                    grid_position = self.board.move_to_location(move)
                    x, y = grid_position
                    print(f"AI가 ({x}, {y})에 돌을 놓습니다.")
                    self.robot.move_to_grid(x, y)
                    self.robot.move_down()
                    self.robot.release_half()
                    self.robot.move_up()
                    self.robot.reload()

                self.human_player.print_board(self.board)

                end, winner = self.board.game_end()
                if end:
                    if winner == 1:
                        print("Game Over! Winner is: Human")
                    elif winner == 2:
                        print("Game Over! Winner is: AI")
                    else:
                        print("Game Over! It's a tie!")
                    break

                current_player = 1 - current_player
            elif key == ord('q'):
                print("Game quit.")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = './Gomoku_AI/save/model_9/policy_9_17100.model'
    game = OmokuGame(model_path)
    game.run()
