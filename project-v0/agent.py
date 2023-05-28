import numpy as np


class DynaQ:
    def __init__(self, states_n, actions_n, alpha, gamma, epsilon):
        self.states_n = states_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.episode = 0
        self.step = 0
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.reward = 0
        self.q_table = np.zeros((self.states_n, self.actions_n))
        self.model = {}
        self.visited_states = {}

    def start_episode(self):
        self.episode += 1
        self.step = 0

    def update(self, state, action, next_state, reward):
        self.__update(state, action, next_state, reward)
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state, action]
        )
        self.model[(state, action)] = (reward, next_state)
        if state in self.visited_states:
            if action not in self.visited_states[state]:
                self.visited_states[state].append(action)
        else:
            self.visited_states[state] = [action]

    def planning_step(self, planning_steps):
        for _ in range(planning_steps):
            state = np.random.choice(list(self.visited_states.keys()))
            action = np.random.choice(self.visited_states[state])
            reward, next_state = self.model[(state, action)]
            self.update(state, action, next_state, reward)

    def __update(self, state, action, next_state, reward):
        self.step += 1
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def get_action(self, state, mode):
        if mode == "random":
            return np.random.choice(self.actions_n)
        elif mode == "greedy":
            return np.argmax(self.q_table[state])
        elif mode == "epsilon-greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self.q_table[state])

    def render(self, mode="step"):
        if mode == "step":
            print(
                f"Episode: {self.episode}, Step: {self.step}, State: {self.state}, ",
                end="",
            )
            print(
                f"Action: {self.action}, Next state: {self.next_state}, Reward: {self.reward}"
            )

        elif mode == "values":
            print(f"Q-Table: {self.q_table}")

class DynaQPlus:
    def __init__(self, states_n, actions_n, alpha, gamma, epsilon, kappa):
        self.states_n = states_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.kappa = kappa
        self.reset()

    def reset(self):
        self.episode = 0
        self.step = 0
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.reward = 0
        self.q_table = np.zeros((self.states_n, self.actions_n))
        self.model = {}
        self.visited_states = {}
        self.tau = np.zeros((self.states_n, self.actions_n))

    def start_episode(self):
        self.episode += 1
        self.step = 0

    def update(self, state, action, next_state, reward):
        self.__update(state, action, next_state, reward)
        # direct RL step
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[state, action]
        )
        
        # update model step
        self.model[(state, action)] = (reward, next_state)
        if state in self.visited_states:
            if action not in self.visited_states[state]:
                self.visited_states[state].append(action)
        else:
            self.visited_states[state] = [action]
        # actions that had never been tried before from a state 
        # should now be allowed to be considered in the planning step
        for a in range(self.actions_n):
            if a not in self.visited_states[state]:
                self.model[(state, a)] = (0, next_state)

        # planning step

    def __update(self, state, action, next_state, reward):
        self.step += 1
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        # update the last-visited counts
        self.tau += 1
        self.tau[state, action] = 0

    def planning_step(self, planning_steps):
        for _ in range(planning_steps):
            state = np.random.choice(list(self.visited_states.keys()))
            action = np.random.choice(list(self.visited_states[state]))
            reward, next_state = self.model[(state, action)]
            reward += self.kappa * np.sqrt(self.tau[state, action])
            self.update(state, action, next_state, reward)
            
    def get_action(self, state, mode):
        if mode == "random":
            return np.random.choice(self.actions_n)
        elif mode == "greedy":
            return np.argmax(self.q_table[state])
        elif mode == "epsilon-greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self.q_table[state])

    def render(self, mode="step"):
        if mode == "step":
            print(
                f"Episode: {self.episode}, Step: {self.step}, State: {self.state}, ",
                end="",
            )
            print(
                f"Action: {self.action}, Next state: {self.next_state}, Reward: {self.reward}"
            )

        elif mode == "values":
            print(f"Q-Table: {self.q_table}")