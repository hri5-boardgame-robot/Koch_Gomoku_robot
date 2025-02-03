import random
import numpy as np
from collections import defaultdict, deque
from game import VanillaBoard, VanillaGame
from mcts_alphaZero import VanillaMCTSPlayer
from policy_value_net import VanillaPolicyValueNet  
from datetime import datetime
import torch
import pickle
import sys
import os
sys.setrecursionlimit(10**8)

class TrainPipeline():
    def __init__(self, use_gpu=True):
        self.board_width, self.board_height = 9, 9
        self.n_in_row = 5
        self.board = VanillaBoard(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = VanillaGame(self.board)

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  
        self.temp = 1.0 
        self.n_playout = 400 
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 512 
        self.play_batch_size = 1
        self.epochs = 5  
        self.kl_targ = 0.02
        self.check_freq = 100  
        self.game_batch_num = 20000  
        self.train_num = 0 
        self.use_gpu = use_gpu  

        self.policy_value_net = VanillaPolicyValueNet(self.board_width, self.board_height)
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Train with GPU")
        else:
            self.device = torch.device('cpu')
            print("Train with CPU")
        self.policy_value_net.to(self.device)  

        self.mcts_player = VanillaMCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = torch.tensor(np.array([data[0] for data in mini_batch]), dtype=torch.float).to(self.device)
        mcts_probs_batch = torch.tensor(np.array([data[1] for data in mini_batch]), dtype=torch.float).to(self.device)
        winner_batch = torch.tensor(np.array([data[2] for data in mini_batch]), dtype=torch.float).to(self.device)

        for i in range(self.epochs):
            with torch.no_grad():
                old_probs, old_v = self.policy_value_net.forward(state_batch)

            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )

            with torch.no_grad():
                new_probs, new_v = self.policy_value_net.forward(state_batch)

            kl = torch.mean(torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), dim=1)).item()
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(f"kl:{kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, loss:{loss:.5f}, entropy:{entropy:.5f}")

        return loss, entropy


    def run(self):
        os.makedirs('./Vanilla_MCTS/save/model_9', exist_ok=True)
        os.makedirs('./Vanilla_MCTS/save/train_9', exist_ok=True)

        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            self.train_num += 1
            print(f"batch i:{self.train_num}, episode_len:{self.episode_len}")

            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()

            if (i + 1) % self.check_freq == 0:
                print(f"★ Save model in {self.train_num}th batch : {datetime.now()}")
                self.policy_value_net.save_model(f'./Vanilla_MCTS/save/model_9/policy_9_{self.train_num}.model')
                with open(f'./Vanilla_MCTS/save/train_9/train_9_{self.train_num}.pickle', 'wb') as f:
                    pickle.dump(self, f)

if __name__ == '__main__':
    print("Train with 9*9 board.")
    train_path = f"./Vanilla_MCTS/save/train_9"
    model_path = f"./Vanilla_MCTS/save/model_9"

    init_num = int(input('Number of trained models stored : '))
    if init_num == 0 or init_num is None:
        training_pipeline = TrainPipeline()
    else:
        training_pipeline = pickle.load(open(f'{train_path}/train_9_{init_num}.pickle', 'rb'))

    print(f"★ Train start : {datetime.now()}")
    training_pipeline.run()
