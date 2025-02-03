import numpy as np

class AlphaBetaPlayer:
    def __init__(self, depth=3):
        self.player = None
        self.opponent = None
        self.depth = depth

    def set_player_ind(self, p):
        self.player = p
        self.opponent = 3 - p

    def heuristic_evaluation(self, board):
        current_player_score = self.evaluate_player_score(board, self.player)
        opponent_score = self.evaluate_player_score(board, self.opponent)
        return current_player_score - opponent_score

    def evaluate_player_score(self, board, player):
        score = 0
        for y in range(board.height):
            for x in range(board.width):
                if board.states_loc[y][x] == player:
                    score += self.evaluate_direction(y, x, board, player)
        return score

    def evaluate_direction(self, row, col, board, player):
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        total_score = 0

        for d in directions:
            player_streak, open_ends_p = self.get_streak_and_openends(row, col, board, player, d)
            opp_streak, open_ends_o = self.get_streak_and_openends(row, col, board, self.opponent, d)

            if player_streak == 2:
                if open_ends_p == 2:
                    total_score += 155  
                else:
                    total_score += 130

            elif player_streak == 3:
                if open_ends_p == 2:
                    total_score += 1100  
                else:
                    total_score += 600

            elif player_streak >= 4:
                total_score += 3000

            if opp_streak >= 4:
                total_score -= 3000
            elif opp_streak == 3:
                if open_ends_o == 2:
                    total_score -= 1100
                else:
                    total_score -= 600

            elif opp_streak == 2:
                if open_ends_o == 2:
                    total_score -= 155
                else:
                    total_score -= 23
        return total_score


    def get_streak_and_openends(self, row, col, board, player, direction):
        dr, dc = direction
        streak = 1

        plus_count = 0
        for k in range(1, 5):
            r = row + dr * k
            c = col + dc * k
            if not (0 <= r < board.height and 0 <= c < board.width):
                break
            if board.states_loc[r][c] == player:
                plus_count += 1
            else:
                break

        minus_count = 0
        for k in range(1, 5):
            r = row - dr * k
            c = col - dc * k
            if not (0 <= r < board.height and 0 <= c < board.width):
                break
            if board.states_loc[r][c] == player:
                minus_count += 1
            else:
                break

        streak = 1 + plus_count + minus_count
        open_ends = 2

        r_plus = row + dr * (plus_count + 1)
        c_plus = col + dc * (plus_count + 1)
        if not (0 <= r_plus < board.height and 0 <= c_plus < board.width):
            open_ends -= 1
        else:
            if board.states_loc[r_plus][c_plus] != 0 and board.states_loc[r_plus][c_plus] != player:
                open_ends -= 1

        r_minus = row - dr * (minus_count + 1)
        c_minus = col - dc * (minus_count + 1)
        if not (0 <= r_minus < board.height and 0 <= c_minus < board.width):
            open_ends -= 1
        else:
            if board.states_loc[r_minus][c_minus] != 0 and board.states_loc[r_minus][c_minus] != player:
                open_ends -= 1

        return streak, open_ends

    def alpha_beta_search(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.game_end()[0]:
            base_evaluation = self.heuristic_evaluation(board)
            return base_evaluation 

        if maximizing_player:
            max_eval = -np.inf
            for move in self.get_available_moves(board):
                board.do_move(move)
                eval_ = self.alpha_beta_search(board, depth - 1, alpha, beta, False)
                board.states.pop(move)
                board.states_loc[move // board.width][move % board.width] = 0
                board.current_player = self.player

                max_eval = max(max_eval, eval_)
                alpha = max(alpha, eval_)
                if beta <= alpha:
                    break
            return max_eval

        else:
            min_eval = np.inf
            for move in self.get_available_moves(board):
                board.do_move(move)
                eval_ = self.alpha_beta_search(board, depth - 1, alpha, beta, True)
                board.states.pop(move)
                board.states_loc[move // board.width][move % board.width] = 0
                board.current_player = self.opponent

                min_eval = min(min_eval, eval_)
                beta = min(beta, eval_)
                if beta <= alpha:
                    break
            return min_eval

    def get_available_moves(self, board):
        available_moves = []
        for y in range(board.height):
            for x in range(board.width):
                if board.states_loc[y][x] == 0:
                    available_moves.append(y * board.width + x) 
        return available_moves


    def detect_threats(self, board):
        threats = []
        for y in range(board.height):
            for x in range(board.width):
                if board.states_loc[y][x] == 0: 
                    for direction in [(1,0), (0,1), (1,1), (1,-1)]:
                        streak, open_ends = self.get_streak_and_openends(y, x, board, self.opponent, direction)
                        if streak >= 4 and open_ends >= 0:  
                            threats.append((y, x))
        return threats

    def get_action(self, board):
        best_move = None
        best_value = -np.inf

        center_y = board.height // 2
        center_x = board.width // 2

        threats = self.detect_threats(board)
        if threats:  
            return threats[0][0] * board.width + threats[0][1]

        sorted_avail = sorted(
            self.get_available_moves(board),
            key=lambda mv: abs((mv // board.width) - center_y) + abs((mv % board.width) - center_x) 
        )

        for move in sorted_avail:
            board.do_move(move)
            move_value = self.alpha_beta_search(board, self.depth - 1, -np.inf, np.inf, False)
            board.states.pop(move)
            board.states_loc[move // board.width][move % board.width] = 0
            board.current_player = self.player

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move

    def simulate_move_evaluation(self, board, move):
        board.do_move(move)
        eval_value = self.heuristic_evaluation(board)
        board.states.pop(move)
        board.states_loc[move // board.width][move % board.width] = 0
        return eval_value

    def __str__(self):
        return f"AlphaBetaPlayer(depth={self.depth})"
