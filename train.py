# train.py
from env import SDWANEnvironment
from agent import DQNAgent
from config import *
from torch.utils.tensorboard import SummaryWriter

env = SDWANEnvironment()
agent = DQNAgent()
writer = SummaryWriter(log_dir="runs/sdwan_latency_control")

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    latencies = []

    for step in range(20):  # arbitrary episode length
        action = agent.select_action(state)
        next_state, reward, done, latency = env.step(action)

        agent.remember(state, action, reward, next_state)
        agent.train_step()
        state = next_state

        total_reward += reward
        latencies.append(latency)

    agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target()

    avg_latency = sum(latencies) / len(latencies)
    writer.add_scalar("Reward/Total", total_reward, episode)
    writer.add_scalar("Latency/Average", avg_latency, episode)
    print(f"Episode {episode+1} - Reward: {total_reward:.2f}, Avg Latency: {avg_latency:.2f}ms")

writer.close()

