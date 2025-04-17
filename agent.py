# agent.py
import torch
import torch.nn.functional as F
import numpy as np
from model import DQN
from replay_buffer import ReplayBuffer
from config import *

class DQNAgent:
    def __init__(self):
        self.q_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer()
        self.epsilon = EPSILON_START
        self.steps = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(ACTION_SIZE)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state).argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        state, action, reward, next_state = self.memory.sample(BATCH_SIZE)
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        q_values = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_state).max(1)[0]
        target_q = reward + GAMMA * next_q

        loss = F.mse_loss(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def remember(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

