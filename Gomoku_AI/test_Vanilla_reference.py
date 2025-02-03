import pygame
import sys
from Vanilla_MCTS.game import VanillaBoard, VanillaGame
from Vanilla_MCTS.mcts_alphaZero import VanillaMCTSPlayer
from Vanilla_MCTS.policy_value_net import VanillaPolicyValueNet
from reference_model import AlphaBetaPlayer
import numpy as np
import time

GRID_SIZE = 9
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
CELL_SIZE = SCREEN_WIDTH // (GRID_SIZE + 1)
LINE_COLOR = (0, 0, 0)
BG_COLOR = (210, 180, 140)
STONE_COLOR_BLACK = (0, 0, 0)
STONE_COLOR_WHITE = (255, 255, 255)
RESTRICTED_COLOR = (0, 150, 0)

restricted_regions = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 0), (2, 8), (8, 3), (8, 4), (8, 5)
]

def draw_final_board(screen, board):
    screen.fill(BG_COLOR)
    for y, x in restricted_regions:
        pygame.draw.rect(
            screen, RESTRICTED_COLOR,
            (
                CELL_SIZE + x * CELL_SIZE - CELL_SIZE // 2,
                CELL_SIZE + y * CELL_SIZE - CELL_SIZE // 2,
                CELL_SIZE,
                CELL_SIZE
            )
        )
    for i in range(GRID_SIZE):
        pygame.draw.line(
            screen, LINE_COLOR,
            (CELL_SIZE, CELL_SIZE + i * CELL_SIZE),
            (CELL_SIZE * GRID_SIZE, CELL_SIZE + i * CELL_SIZE)
        )
        pygame.draw.line(
            screen, LINE_COLOR,
            (CELL_SIZE + i * CELL_SIZE, CELL_SIZE),
            (CELL_SIZE + i * CELL_SIZE, CELL_SIZE * GRID_SIZE)
        )
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            loc_move = board.location_to_move([y, x])
            player_val = board.states.get(loc_move, 0)
            if player_val == 1:  
                pygame.draw.circle(
                    screen, STONE_COLOR_BLACK,
                    (CELL_SIZE + x * CELL_SIZE, CELL_SIZE + y * CELL_SIZE),
                    CELL_SIZE // 3
                )
            elif player_val == 2:  
                pygame.draw.circle(
                    screen, STONE_COLOR_WHITE,
                    (CELL_SIZE + x * CELL_SIZE, CELL_SIZE + y * CELL_SIZE),
                    CELL_SIZE // 3
                )
    pygame.display.flip()

def test_model_multiple_positions(model_path, board_width=9, board_height=9, n_in_row=5):
    black_moves = [
        (3, 3), (3, 4), (3, 5),
        (4, 3), (4, 4), (4, 5),
        (5, 3), (5, 4), (5, 5)
    ]

    results = []

    for idx, black_move in enumerate(black_moves, start=1):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Simulation {idx}")
    
        board = VanillaBoard(width=board_width, height=board_height, n_in_row=n_in_row)
        game = VanillaGame(board)

        policy_value_net1 = VanillaPolicyValueNet(board_width, board_height)
        policy_value_net1.load_model(model_path)
        ai_player_2 = VanillaMCTSPlayer(
            policy_value_net1.policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0
        )

        ai_player_1 = AlphaBetaPlayer(depth=1) 

        board.init_board(start_player=0)

        if not hasattr(board, 'availables'):
            def availables(self):
                return list(set(range(self.width * self.height)) - set(self.states.keys()))
            setattr(board.__class__, 'availables', property(availables))

        black_move_index = board.location_to_move(black_move)
        board.current_player = 1
        board.do_move(black_move_index)

        board.current_player = 2

        winner = game.start_play(
            ai_player_1,  
            ai_player_2,  
            start_player=1,
            is_shown=0,
            skip_init=True 
        )

        draw_final_board(screen, board)
        print(f"\nSimulation {idx} : First position of Black = {black_move}, Winner = {winner}")

        start_time = time.time()
        show_duration = 2
        running = True
        while running:
            current_time = time.time()
            if current_time - start_time > show_duration:
                running = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.flip()

        if winner == 1:
            results.append((black_move, "Black"))
        elif winner == 2:
            results.append((black_move, "White"))
        else:
            results.append((black_move, "Tie"))

        pygame.quit()

    print("\n--- Simulation Results ---")
    for i, (pos, res) in enumerate(results, start=1):
        print(f"Simulation {i}: Black's first move={pos}, Winner={res}")

if __name__ == '__main__':
    model_path = './Vanilla_MCTS/save/model_9/Vanilla_MCTS.model'
    test_model_multiple_positions(model_path)
