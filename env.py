# env.py
import numpy as np

class SDWANEnvironment:
    def __init__(self):
        self.num_paths = 4
        self.num_metrics = 3
        self.state_size = self.num_paths * self.num_metrics
        self.reset()

    def reset(self):
        self.paths = self._generate_paths()
        return self._get_state()

    def _generate_paths(self):
        paths = []
        for _ in range(self.num_paths):
            latency = np.random.uniform(50, 150)
            jitter = np.random.uniform(0, 10)
            loss = np.random.uniform(0, 1)
            paths.append([latency, jitter, loss])
        return np.array(paths)

    def _get_state(self):
        return self.paths.flatten()

    def step(self, action):
        selected_path = self.paths[action]
        latency = selected_path[0]
        reward = -latency / 100.0
        self.paths = self._generate_paths()
        next_state = self._get_state()
        done = False
        return next_state, reward, done, latency
