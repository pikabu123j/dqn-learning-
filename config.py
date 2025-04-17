# config.py

STATE_SIZE = 12      # 4 paths × 3 metrics (latency, jitter, loss)
ACTION_SIZE = 4      # 4 possible paths
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
