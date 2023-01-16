import gym
import gym_environments
from agent import TwoArmedBandit

env = gym.make('TwoArmedBandit-v0')
agent = TwoArmedBandit(0.1, 0.1)  # alpha, epsilon

env.reset()

total_reward = [0, 0]

for iteration in range(100):
    action = agent.get_action("epsilon-greedy")    
    _, reward, _, _, _ = env.step(action)
    agent.update(action, reward) 
    agent.render()
    # performance measure
    total_reward[action] += reward

print(f"Total Rewards: {total_reward}, Best Action: {total_reward.index(max(total_reward))}")

env.close()