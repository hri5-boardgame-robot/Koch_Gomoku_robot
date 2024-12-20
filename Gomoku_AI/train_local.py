import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet  # PyTorch로 변경된 네트워크
from datetime import datetime
import torch
import pickle
import sys
import os
sys.setrecursionlimit(10**8)

class TrainPipeline():
    def __init__(self, use_gpu=True):
        # 게임(오목)에 대한 변수들
        self.board_width, self.board_height = 9, 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # 학습에 대한 변수들
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # KL에 기반하여 학습 계수를 적응적으로 조정
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 512  # mini-batch size
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 100  # 모델 체크 및 저장 빈도
        self.game_batch_num = 50000  # 최대 학습 횟수
        self.train_num = 0 # 현재 학습 횟수
        self.use_gpu = use_gpu  # GPU 사용 여부

        # PyTorch 기반의 policy-value net 초기화
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("GPU 사용 설정 완료")
        else:
            self.device = torch.device('cpu')
            print("CPU 사용 설정 완료")
        self.policy_value_net.to(self.device)  # 모델을 GPU/CPU로 전송

        # Monte Carlo Tree Search 플레이어 설정
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        회전 및 뒤집기로 데이터셋 확대
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 반시계 방향으로 회전
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 수평으로 뒤집기
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """학습을 위한 자가 플레이 데이터를 수집"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 데이터를 확대
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """정책-가치 네트워크 업데이트"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = torch.tensor(np.array([data[0] for data in mini_batch]), dtype=torch.float).to(self.device)
        mcts_probs_batch = torch.tensor(np.array([data[1] for data in mini_batch]), dtype=torch.float).to(self.device)
        winner_batch = torch.tensor(np.array([data[2] for data in mini_batch]), dtype=torch.float).to(self.device)

        for i in range(self.epochs):
            # KL divergence를 계산하여 학습률 조정
            with torch.no_grad():
                old_probs, old_v = self.policy_value_net.forward(state_batch)

            # 모델 업데이트 수행
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )

            with torch.no_grad():
                new_probs, new_v = self.policy_value_net.forward(state_batch)

            kl = torch.mean(torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), dim=1)).item()
            if kl > self.kl_targ * 4:
                break

        # 학습률 조정
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(f"kl:{kl:.5f}, lr_multiplier:{self.lr_multiplier:.3f}, loss:{loss:.5f}, entropy:{entropy:.5f}")

        return loss, entropy


    def run(self):
        # 모델과 학습 파이프라인의 저장 폴더가 존재하지 않는 경우 생성
        os.makedirs('./save/model_9', exist_ok=True)
        os.makedirs('./save/train_9', exist_ok=True)

        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            self.train_num += 1
            print(f"batch i:{self.train_num}, episode_len:{self.episode_len}")

            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()

            # 현재 모델의 성능 체크 및 저장
            if (i + 1) % self.check_freq == 0:
                print(f"★ {self.train_num}번째 batch에서 모델 저장 : {datetime.now()}")
                self.policy_value_net.save_model(f'./save/model_9/policy_9_{self.train_num}.model')
                with open(f'./save/train_9/train_9_{self.train_num}.pickle', 'wb') as f:
                    pickle.dump(self, f)

if __name__ == '__main__':
    print("9x9 환경에서 학습을 진행합니다.")
    train_path = f"./save/train_9"
    model_path = f"./save/model_9"

    init_num = int(input('현재까지 저장된 모델의 학습 수 : '))
    if init_num == 0 or init_num is None:
        training_pipeline = TrainPipeline()
    else:
        training_pipeline = pickle.load(open(f'{train_path}/train_9_{init_num}.pickle', 'rb'))

    print(f"★ 학습시작 : {datetime.now()}")
    training_pipeline.run()
