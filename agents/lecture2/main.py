import os

import gym
import gym_environments
import time
from agent import ValueIteration, PolicyIteration

# Allowing environment to have sounds
if "SDL_AUDIODRIVER" in os.environ:
    del os.environ["SDL_AUDIODRIVER"]

# RobotBattery-v0, FrozenLake-v1, FrozenLake-v2
env = gym.make('FrozenLake-v2', render_mode="human")
# agent = PolicyIteration(env.observation_space.n, env.action_space.n, env.P, 0.8)
agent = ValueIteration(env.observation_space.n, env.action_space.n, env.P, 0.8)

agent.solve(100) # value iteration
# agent.solve() # policy iteration

agent.render()

total_reward = 0
num_tests = 100

for i in range(num_tests):
    print(f"Test {i}")
    observation, info = env.reset()
    terminated, truncated = False, False

    env.render()
    # time.sleep(2)

    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
    
    total_reward += reward
    
    # time.sleep(2)
env.close()

print (f"Average Total Reward: {total_reward/num_tests}\n")