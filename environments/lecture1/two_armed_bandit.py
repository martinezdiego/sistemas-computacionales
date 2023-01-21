from gym.envs.registration import register
import gym

from agent import TwoArmedBandit

register(
    id="TwoArmedBandit-v1",
    entry_point="environment:TwoArmedBanditEnv"
)

env = gym.make("TwoArmedBandit-v1")
agent = TwoArmedBandit(0.1, 0.3)

env.reset(seed=31)

for _ in range(100):
    action = agent.get_action("epsilon-greedy")
    _, reward, _, _, _ = env.step(action)
    agent.update(action, reward)
    agent.render()


