import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1  # Player 1 starts
        return self.board.flatten()

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")

        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            raise ValueError("Invalid move")
        
        self.board[row, col] = self.current_player

        if self.check_winner(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1
        elif (self.board == 0).sum() == 0:
            self.done = True
            self.winner = 0
            reward = 0.5  # Draw
        else:
            self.current_player = -self.current_player
            reward = 0

        return self.board.flatten(), reward, self.done, {}

    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import random

def train(env, q_network, episodes=5000, gamma=0.99, epsilon=0.1, lr=0.001):
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for episode in range(episodes):
        state = torch.FloatTensor(env.reset()).unsqueeze(0)
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(np.where(env.board.flatten() == 0)[0])
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = q_values.argmax().item()
                    if env.board.flatten()[action] != 0:  # invalid action, choose randomly
                        action = random.choice(np.where(env.board.flatten() == 0)[0])
            
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.FloatTensor([reward])

            with torch.no_grad():
                next_q_values = q_network(next_state)
                max_next_q_value = next_q_values.max()

            target_q_value = reward + gamma * max_next_q_value
            predicted_q_value = q_network(state)[0, action]

            loss = loss_fn(predicted_q_value, target_q_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode} - Loss: {loss.item()}")

env = TicTacToe()
q_network = QNetwork()
train(env, q_network)

def play_game(env, q_network):
    state = torch.FloatTensor(env.reset()).unsqueeze(0)
    done = False
    while not done:
        with torch.no_grad():
            q_values = q_network(state)
            action = q_values.argmax().item()
            if env.board.flatten()[action] != 0:
                action = random.choice(np.where(env.board.flatten() == 0)[0])
        next_state, reward, done, _ = env.step(action)
        state = torch.FloatTensor(next_state).unsqueeze(0)
        print(env.board)
        print()

env = TicTacToe()
play_game(env, q_network)
