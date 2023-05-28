import sys
import gym

from gym.envs.registration import register

from agent import DynaQ, DynaQPlus

register(
    id="farm-v0",
    entry_point="farm:FarmEnv"
)

def run(env, agent, selection_method, episodes, planning_steps):
    for episode in range(episodes):
        if episode > 0:
            print(f"Episode: {episode+1}")
        observation, _ = env.reset()
        agent.start_episode()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, selection_method)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, next_observation, reward)
            observation = next_observation
        if selection_method == "epsilon-greedy":
            # for _ in range(100):
            #     state = np.random.choice(list(agent.visited_states.keys()))
            #     action = np.random.choice(agent.visited_states[state])
            #     reward, next_state = agent.model[(state, action)]
            #     agent.update(state, action, next_state, reward)
            agent.planning_step(planning_steps)

if __name__ == "__main__":
    episodes = 350 if len(sys.argv) < 2 else int(sys.argv[1])
    env = gym.make("farm-v0")
    agent = DynaQ(env.observation_space.n, env.action_space.n, alpha=1.0, gamma=0.95, epsilon=0.1)
    # train
    run(env, agent, "epsilon-greedy", episodes, 0)
    agent.render()
    # play
    env = gym.make("farm-v0", render_mode="human")
    run(env, agent, "greedy", 1, 0)
    agent.render()
    env.close()