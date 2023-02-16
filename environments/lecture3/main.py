from gym.envs.registration import register
import gym

from agent import MonteCarlo

register(
    id="FrozenLake-v3",
    entry_point="frozen_lake:FrozenLake"
)

def train(env, agent, episodes):
    for episode in range(episodes):
        print(episode)
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation)
            new_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, reward, terminated)
            observation = new_observation

def play(env, agent):
    observation, _ = env.reset()
    terminated, truncated = False, False

    env.render()

    while not (terminated or truncated):
        action = agent.get_best_action(observation)
        observation, _, terminated, truncated, _ = env.step(action)

env = gym.make("FrozenLake-v3", rows=10, cols=10)
agent = MonteCarlo(env.observation_space.n, env.action_space.n, gamma=0.9, epsilon=0.9)
train(env, agent, episodes=2000)
agent.render()
env.init_render_mode("human")
play(env, agent)

env.close()