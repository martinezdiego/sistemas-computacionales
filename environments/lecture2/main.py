import os
import time

from gym.envs.registration import register
import gym

from agent import PolicyIteration as PolicyIterationAgent

if "SDL_AUDIODRIVER" in os.environ:
    del os.environ["SDL_AUDIODRIVER"]

register(
    id="RobotBattery-v2",
    entry_point="robot_battery:RobotBatteryEnv"
)

env = gym.make("RobotBattery-v2", render_mode="human")
agent = PolicyIterationAgent(env.observation_space.n, env.action_space.n, env.P, 0.9)

agent.solve()
# agent.render()

num_tests = 5

for _ in range(num_tests):
    observation, info = env.reset()
    terminated, truncated = False, False

    env.render()
    time.sleep(2)

    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

    time.sleep(2)

env.close()