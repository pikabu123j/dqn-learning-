# replay_buffer.py
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.array, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)
