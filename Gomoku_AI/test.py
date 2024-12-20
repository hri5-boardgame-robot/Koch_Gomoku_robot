import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet  # 학습된 모델을 불러옵니다.

def test_model(model_path, board_width=9, board_height=9, n_in_row=5):
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    game = Game(board)

    policy_value_net = PolicyValueNet(board_width, board_height)
    policy_value_net.load_model(model_path)

    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)

    human_player = HumanPlayer()  

    start_player = 0  
    winner = game.start_play(human_player, mcts_player, start_player=start_player, is_shown=0)  

    if winner == 1:
        print("Game Over! Winner is: Human")
    elif winner == 2:
        print("Game Over! Winner is: AI")
    else:
        print("Game Over! It's a tie!")

class HumanPlayer(object):
    """사용자가 직접 플레이어로 게임에 참여할 수 있도록 입력 받는 클래스"""
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            self.print_board(board)
            move = input("당신의 차례입니다. 착수 위치 (ex: 2,3)를 입력하세요: ")
            h, w = map(int, move.split(","))
            move = board.location_to_move([h, w])
            if move in board.states:
                print("이미 돌이 놓인 위치입니다. 다시 입력하세요.")
                return self.get_action(board)
            return move
        except Exception as e:
            print("잘못된 입력입니다. 다시 입력하세요.")
            return self.get_action(board)

    def print_board(self, board):
        """바둑판을 터미널에 보기 좋게 출력"""
        size = board.width
        print("   " + "   ".join(map(str, range(size))))  
        for i in range(size):
            row = ["." if (i * size + j) not in board.states else ("X" if board.states[i * size + j] == 1 else "O") for j in range(size)]
            print(f"{i}  " + "   ".join(row)) 

    def __str__(self):
        return "HumanPlayer"

if __name__ == '__main__':
    model_path = './save/model_9/policy_9_17100.model'
    test_model(model_path)
