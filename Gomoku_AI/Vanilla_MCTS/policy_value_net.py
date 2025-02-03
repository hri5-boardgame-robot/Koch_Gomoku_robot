import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class VanillaPolicyValueNet(nn.Module):
    def __init__(self, board_width, board_height):
        super(VanillaPolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # Convolutional layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Action policy layers
        self.policy_conv = nn.Conv2d(128, 5, kernel_size=1)
        self.policy_fc = nn.Linear(5 * board_width * board_height, board_width * board_height)

        # State value layers
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        state_input = state_input.to(next(self.parameters()).device)
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        policy_x = F.relu(self.policy_conv(x))
        policy_x = policy_x.view(policy_x.size(0), -1)
        action_probs = F.softmax(self.policy_fc(policy_x), dim=1)

        # Value head
        value_x = F.relu(self.value_conv(x))
        value_x = value_x.view(value_x.size(0), -1)
        value_x = F.relu(self.value_fc1(value_x))
        state_value = torch.tanh(self.value_fc2(value_x))

        return action_probs, state_value

    def policy_value_fn(self, board):
        legal_positions = list(set(range(self.board_width * self.board_height)) - set(board.states.keys()))
        current_state = board.current_state()
        current_state = np.ascontiguousarray(current_state)
        state_input = torch.from_numpy(current_state).float().unsqueeze(0).to(next(self.parameters()).device)
        action_probs, value = self.forward(state_input)
        action_probs = action_probs.detach().cpu().numpy().flatten() 
        act_probs = zip(legal_positions, action_probs[legal_positions])
        return act_probs, value.item()


    def train_step(self, state_batch, mcts_probs_batch, winner_batch, learning_rate):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)

        action_probs, value = self.forward(state_batch)

        # Loss calculation
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * torch.log(action_probs + 1e-10), dim=1))
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))
        return loss.item(), entropy.item()

    
    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        print(f"Model is saved in {model_path}.")

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print(f"Model is loaded in {model_path}.")
