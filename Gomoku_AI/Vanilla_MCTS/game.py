# -*- coding: utf-8 -*-
import numpy as np
from .renju_rule import Renju_Rule
from IPython.display import clear_output
import os


class VanillaBoard(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  
        self.forbidden_moves_ai = []

    def init_board(self, start_player=0):
        self.order = start_player
        self.current_player = self.players[start_player]
        self.last_move, self.last_loc = -1, -1

        self.states, self.states_loc = {}, np.array([
            [0] * self.width for _ in range(self.height)])
        self.forbidden_locations, self.forbidden_moves = [], []

    def move_to_location(self, move): 
        h = move // self.width  
        w = move % self.width  
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h, w = location[0], location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        square_state = np.zeros((5, self.width, self.height)) 
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0

        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  

        hardware_restricted = [
            (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8),
            (1,0), (1,1), (1,2), (1,3), (1,5), (1,6), (1,7), (1,8),
            (2,0), (2,8),
            (8,3), (8,4), (8,5)
        ]

        for r, c in hardware_restricted:
            if 0 <= r < self.width and 0 <= c < self.height:
                square_state[4][r, c] = 1.0

        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        loc = self.move_to_location(move)
        self.states_loc[loc[0]][loc[1]] = 1 if self.is_you_black() else 2
        self.current_player = (
            self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move, self.last_loc = move, loc

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(self.states.keys())
        if len(moved) < self.n_in_row * 2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif len(self.states) == self.width*self.height:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def set_forbidden(self):
        rule = Renju_Rule(self.states_loc, self.width)
        if self.order == 0:
            self.forbidden_locations = rule.get_forbidden_points(stone=1)
        else:
            self.forbidden_locations = rule.get_forbidden_points(stone=2)
        
        # Region of hardward restrict
        hardware_restricted = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (1,0), (1,1), (1,2), (1,3), (1,5), (1,6), (1,7), (1,8), (2,0), (2,8), (8,3), (8,4), (8,5)]
        self.forbidden_moves_ai = [self.location_to_move(loc) for loc in hardware_restricted]
        
        self.forbidden_moves = [self.location_to_move(loc) for loc in self.forbidden_locations]

    def is_you_black(self):
        if self.order == 0 and self.current_player == 1:
            return True
        elif self.order == 1 and self.current_player == 2:
            return True
        else:
            return False


class VanillaGame(object):
    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        width = board.width
        height = board.height

        clear_output(wait=True)
        os.system('cls')

        print()
        if board.order == 0 : 
            print("Black(●) : Player")
            print("White(○) : AI")
        else :
            print("Black(●) : AI")
            print("White(○) : Player")
        print("--------------------------------\n")
        
        if board.current_player == 1 : print("Your turn.\n")
        else : print("Ai is thinking...\n")
            
        row_number = ['⒪','⑴','⑵','⑶','⑷','⑸','⑹','⑺','⑻','⑼','⑽','⑾','⑿','⒀','⒁']
        print('　', end='')
        for i in range(height):
            print(row_number[i], end='')
        print()
        for i in range(height):
            print(row_number[i], end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('●' if board.order == 0 else '○', end='')
                elif p == player2:
                    print('○' if board.order == 0 else '●', end='')
                elif board.is_you_black() and (i, j) in board.forbidden_locations:
                    print('Ⅹ', end='')
                else:
                    print('　', end='')
            print()
        if board.last_loc != -1:
            print(f"Last position : ({board.last_loc[0]},{board.last_loc[1]})\n")

    def start_play(self, player1, player2, start_player=0, is_shown=1, skip_init = False):
        
        if not skip_init : 
            self.board.init_board(start_player)

        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        forbidden_count1 = 0
        forbidden_count2 = 0  
        total_moves1 = 0  
        total_moves2 = 0

        while True:
            if self.board.is_you_black():
                self.board.set_forbidden()

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)

            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]

            move = player_in_turn.get_action(self.board)

            if move in self.board.forbidden_moves_ai:
                if current_player == 1:
                    forbidden_count1 += 1
                else:  # current_player == 2
                    forbidden_count2 += 1

            if current_player == 1:
                total_moves1 += 1
            else:
                total_moves2 += 1

            self.board.do_move(move)

            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")

                if total_moves1 > 0:
                    forbidden_ratio1 = forbidden_count1 / total_moves1
                    print(f"Forbidden move ratio (AI_player_1): {forbidden_ratio1:.2%}")

                if total_moves2 > 0:
                    forbidden_ratio2 = forbidden_count2 / total_moves2
                    print(f"Forbidden move ratio (AI_player_2): {forbidden_ratio2:.2%}")

                return winner


    def start_self_play(self, player, is_shown=0, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            if self.board.is_you_black():
                self.board.set_forbidden()
            if is_shown:
                self.graphic(self.board, p1, p2)

            move, move_probs = player.get_action(
                self.board, temp=temp, return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            self.board.do_move(move)

            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                if is_shown:
                    self.graphic(self.board, p1, p2)
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
