import numpy as np
from game import VanillaBoard, VanillaGame
from mcts_alphaZero import VanillaMCTSPlayer
from policy_value_net import VanillaPolicyValueNet 

def test_model(model_path, board_width=9, board_height=9, n_in_row=5):
    board = VanillaBoard(width=board_width, height=board_height, n_in_row=n_in_row)
    game = VanillaGame(board)

    policy_value_net = VanillaPolicyValueNet(board_width, board_height)
    policy_value_net.load_model(model_path)

    mcts_player = VanillaMCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0)

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
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            self.print_board(board)
            move = input("Your turn. Enter your move (e.g., 2,3): ")
            h, w = map(int, move.split(","))
            move = board.location_to_move([h, w])
            if move in board.states:
                print("The position is already occupied. Please enter again.")
                return self.get_action(board)
            return move
        except Exception as e:
            print("Invalid input. Please try again.")
            return self.get_action(board)

    def print_board(self, board):
        size = board.width
        print("   " + "   ".join(map(str, range(size))))  
        for i in range(size):
            row = ["." if (i * size + j) not in board.states else ("X" if board.states[i * size + j] == 1 else "O") for j in range(size)]
            print(f"{i}  " + "   ".join(row)) 

    def __str__(self):
        return "HumanPlayer"

if __name__ == '__main__':
    model_path = '' # Path of your model.
    test_model(model_path)
