import sys
import time
import gym
import gym_environments
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import DynaQ, DynaQPlus


def run(env, agent, selection_method, episodes, planning_steps, steps_per_episode):
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
        steps_per_episode[episode] = agent.step
        if selection_method == "epsilon-greedy":
            # for _ in range(100):
            #     state = np.random.choice(list(agent.visited_states.keys()))
            #     action = np.random.choice(agent.visited_states[state])
            #     reward, next_state = agent.model[(state, action)]
            #     agent.update(state, action, next_state, reward)
            agent.planning_step(planning_steps)

def run_experiment(env, agent, params : dict):

    num_runs = params.get('num_runs', 1)
    num_episodes = params.get('num_episodes', 350)
    planning_steps = params.get('planning_steps', [100])
    selection_method = params.get('selection_method', 'epsilon-greedy')           
    steps_per_episode = np.zeros((len(planning_steps), num_runs, num_episodes))
    data = {
        'num_runs': num_runs, 
        'num_episodes': num_episodes,
        'planning_steps' : planning_steps,
        'step_size': agent.alpha,
        'discount': agent.gamma,
        'epsilon': agent.epsilon,
        'kappa': agent.kappa if (hasattr(agent, 'kappa')) else 0
    }

    for i, planning_steps in enumerate(planning_steps):
        print('Planning steps : ', planning_steps)
        time.sleep(0.5)

        for j in tqdm(range(num_runs)):
            run(env, agent, selection_method, num_episodes, planning_steps, steps_per_episode[i][j])

    data['steps_per_episode'] = steps_per_episode
    np.save(f"{agent.__class__.__name__}_{agent.alpha}_{agent.epsilon}", data)
    
def plot_data(filename_dynaq, filename_dynaqplus):

    dynaq_data = np.load(filename_dynaq, allow_pickle=True).item()
    # dynaqplus_data = np.load(filename_dynaqplus, allow_pickle=True).item()

    # only first amount of planning steps
    steps_per_episode_dynaq = dynaq_data['steps_per_episode'][0]
    # steps_per_episode_dynaqplus = dynaqplus_data['steps_per_episode'][0]
    planning_steps = dynaq_data['planning_steps'][0]
    num_runs = dynaq_data['num_runs']

    alpha = dynaq_data['step_size']
    epsilon = dynaq_data['epsilon']
    gamma = dynaq_data['discount']
    # kappa = dynaqplus_data['kappa']

    # plt.plot(np.mean(steps_per_episode_dynaqplus, axis=0), label='Dyna-Q+')
    plt.plot(np.mean(steps_per_episode_dynaq, axis=0), label='Dyna-Q' )
    plt.xlabel('Episodes')
    plt.ylabel('Steps\nper\nepisode', rotation=0, labelpad=30)
    plt.legend(loc='upper right')
    plt.title('Average performance of Dyna-Q agents in Blocks-v0 environment\n' + 
              r'$\epsilon$=' + f'{epsilon} ' + 
              r'$\alpha$=' + f'{alpha} ' +
              r'$\gamma$=' + f'{gamma} ' + 
              # r'$\kappa$=' + f'{kappa} \n' +
              f'planning steps={planning_steps} ' +
              f'runs={1}')
    plt.show()

if __name__ == "__main__":
    environments = ["Princess-v0", "Blocks-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 350 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id])
    
    agent = DynaQ(
        env.observation_space.n, env.action_space.n, alpha=1.0, gamma=0.95, epsilon=0.1
    )
    agent2 = DynaQPlus(        
        env.observation_space.n, env.action_space.n, alpha=1.0, gamma=0.95, epsilon=0.1, kappa=0.001
    )
    params = {
        'num_runs': 1,
        'num_episodes': episodes,
        'planning_steps': [50],
        'selection_method': 'epsilon-greedy'
    }
    
    # Train
    #run(env, agent, "epsilon-greedy", episodes)
    # run_experiment(env, agent, params)
    # run_experiment(env, agent2, params)

    env.close()

    # Plot
    plot_data("results/runs_1/DynaQ_1.0_0.8.npy", "results/runs_1/DynaQPlus_1.0_0.4.npy")

    # Play
    # env = gym.make(environments[id], render_mode="human")
    # run(env, agent, "greedy", 1, 0, [0])
    # agent.render()
    # env.close()
