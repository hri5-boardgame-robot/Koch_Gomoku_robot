from utils import *
from initial import *
from initial import manual_warping, find_board
import cv2
from enum import Enum

PLAYER = {
	1: "BLACK",
	2: "WHITE"
}

class OmokModel:
	def __init__(self, board_size=9):
		self.board_size = board_size

	def update_board(self, row, col, player):
		# player : 흑이냐 백이냐에 따라 달라짐 (흑이면 1, 백이면 2)
		if 0 <= row < self.board_size and 0 <= col < self.board_size:
			self.board[row, col] = player

	def initialize(self, edges):
		self.board = np.zeros((self.board_size, self.board_size), dtype=int)
		tl, br = edges
		self.grid_points = get_grid_points(tl, br, size=self.board_size)

	def get_board_state(self):
		return self.board

	def reset_board(self):
		self.board.fill(0)


class OmokView: # 이미지 프로세싱 처리 하는 곳 (관찰 => 이미치 처리 => STATE UPDATE SIGNAL 보내)
	def __init__(self, model):
		"""
		Initialize the Omok view.
		:param model: An instance of OmokModel.
		"""
		self.model = model
		self.cell_size = 40  # Size of each cell in pixels for visualization
		self.margin = 20  # Margin around the board
		self.board_image = self._create_board_image()
		self.edges = None

	def _create_board_image(self):
		"""
		Create a blank board image for visualization.
		:return: A blank board image.
		"""
		size = self.model.board_size * self.cell_size + 2 * self.margin
		board = np.ones((size, size, 3), dtype=np.uint8) * 255  # White background
		for i in range(self.model.board_size + 1):
			x = self.margin + i * self.cell_size
			cv2.line(board, (x, self.margin), (x, size - self.margin), (0, 0, 0), 1)
			cv2.line(board, (self.margin, x), (size - self.margin, x), (0, 0, 0), 1)
		return board

	def render(self, frame):
		"""
		Render the current board state on top of the camera frame.
		:param frame: The live camera feed frame.
		"""
		overlay = self.board_image.copy()

		for row in range(self.model.board_size):
			for col in range(self.model.board_size):
				state = self.model.board[row, col]
				if state == 1:  # Player 1
					center = (self.margin + col * self.cell_size, self.margin + row * self.cell_size)
					cv2.circle(overlay, center, self.cell_size // 3, (0, 0, 255), -1)  # Red circle
				elif state == 2:  # Player 2
					center = (self.margin + col * self.cell_size, self.margin + row * self.cell_size)
					cv2.circle(overlay, center, self.cell_size // 3, (255, 0, 0), -1)  # Blue circle

		# Combine the camera frame and the board overlay
		combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
		cv2.imshow("Omok Board State Checker", combined)


### Controller ###
class OmokController:
	def __init__(self, model, view):
		"""
		Initialize the Omok controller.
		:param model: An instance of OmokModel.
		:param view: An instance of OmokView.
		"""
		self.model = model
		self.view = view
		self.current_player = 1  # Start with Player 1(흑돌)
		

	# def process_frame(self, frame):
	#     """
	#     Process each camera frame (e.g., detect moves, update the board).
	#     :param frame: The current camera frame.
	#     """

	#     row, col, player = move
	#     self.model.update_board(row, col, player)

	def run(self):
		"""
		Run the Omok state checker.
		"""
		H = None
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			print("Error: Unable to access the camera.")
			return

		print("Press 'c' to capture a frame, or 'q' to quit.")

		ret, frame_prev = cap.read()
		if not ret:
			print("Error: Unable to capture the first frame.")
			return
		line_window =False		
		while True:
			ret, frame = cap.read()
			frame_curr = frame.copy()
			if not ret:
				print("Error: Unable to read from the camera.")
				break

			if H is None:
				cv2.imshow("Ready State", frame)
			else:
				#cv2.destroyWindow('Ready State')
				frame_curr = warp_planar(frame_curr, H, (450,450))
				cv2.imshow("While Game", frame_curr)
			key = cv2.waitKey(1) & 0xFF

			if key == ord('q'):
				print("quit game")
				break
			
			if key == ord('c'):  # 카메라 조정 후, 시작해서 homography 진행
				cv2.waitKey(0)
				print("ploting 4 point")
				warped_image, H = manual_warping(frame)
				res = dict()
				edges = find_board(warped_image, res, size=9) # TODO input을 model로 변경
				self.model.initialize(edges=edges)

				# # NOTE debugging
				cv2.imshow("line detection", res[GR_IMG_LINES2])
				# line_window = True

			elif key == ord('n'): # 돌을 놓고 눌러야함
				print(f"next turn: {PLAYER[self.current_player]}")
				# 현재 프레임으로 받으면 안되고, homo warping 한 프레임을 받아야 함
				
				# update_point가 현재 업데이트 된 돌의 위치
				update_point, circle = update_board_circle(curr=frame_curr, model=self.model, player=self.current_player)
				cv2.imshow("draw unocc circle", cv2.circle(frame_curr, circle[:2], circle[2],(0, 255, 0), 2))
				self.current_player = 3 - self.current_player # change player
				

			frame_prev = frame_curr
		cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	model = OmokModel()
	view = OmokView(model)
	controller = OmokController(model, view)
	controller.run()