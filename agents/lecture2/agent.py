import numpy as np


class ValueIteration():
    def __init__(self, states_n, actions_n, P, gamma):
        self.states_n = states_n
        self.actions_n = actions_n
        self.P = P
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.values = np.zeros(self.states_n)
        self.policy = np.zeros(self.states_n)

    def get_action(self, state):
        return int(self.policy[state])

    def render(self):
        print("Values: {}, Policy: {}".format(self.values, self.policy))

    def solve(self, iterations):
        for _ in range(iterations):
            for s in range(self.states_n):
                values = [sum([prob * (r + self.gamma * self.values[s_])
                               for prob, s_, r, _ in self.P[s][a]])
                          for a in range(self.actions_n)]
                self.values[s] = max(values)
                self.policy[s] = np.argmax(np.array(values))


class PolicyIteration():
    def __init__(self, states_n, actions_n, P, gamma=1, epsilon=1e-10):
        self.states_n = states_n
        self.actions_n = actions_n
        self.P = P
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.values = np.zeros(self.states_n)
        self.policy = np.zeros(self.states_n)

    def get_action(self, state):
        return int(self.policy[state])
    
    def render(self):
        print("Values: {}, Policy: {}".format(self.values, self.policy))
    
    def Q(self, state, action):
        return sum([prob * (reward + self.gamma * self.values[new_state])
                        for prob, new_state, reward, _ in self.P[state][action]])

    def solve(self):
        # iterative algorithm of policy evaluation
        while True:
            # policy evaluation loop
            while True:
                new_values = [ self.Q(state, self.policy[state]) for state in range(self.states_n) ]
                # check for convergence
                if max(abs(self.values[state] - new_values[state]) for state in range(self.states_n)) < self.epsilon:
                    break
                self.values = new_values
            # policy improvement
            policy_stable = True
            for state in range(self.states_n):
                old_action = self.policy[state]
                self.policy[state] = np.argmax([ self.Q(state, action) for action in range(self.actions_n) ])
                if old_action != self.policy[state]:
                    policy_stable = False
            if policy_stable:
                break
