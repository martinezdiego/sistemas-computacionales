import time
import numpy as np
import gym

import settings
from world import World

class RobotBatteryEnv(gym.Env):
    
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(settings.NUM_ACTIONS)
        self.observation_space = gym.spaces.Discrete(settings.NUM_TILES)
        self.action = 0
        self.reward = 0.0
        self.state = np.random.randint(0, settings.NUM_TILES)
        self.current_battery = settings.BATTERY_LOAD
        self.delay = settings.DEFAULT_DELAY
        self.finish_state = self.__get_state(np.random.randint(0, settings.ROWS), np.random.randint(0, settings.COLS))
        self.win = 0
        self.lose = 0
        self.__init_P()
        self.world = World(
            "Robot Battery Environment",
            self.state,
            self.action,
            self.finish_state
        )

        self.reset()

    def __init_P(self):
        self.P = { state : { action: [] for action in range(settings.NUM_ACTIONS) } for state in range(settings.NUM_TILES) }
        for row in range(settings.ROWS):
            for col in range(settings.COLS):
                state = self.__get_state(row, col)
                for action in range(settings.NUM_ACTIONS):
                    if (action == 0):
                        to_state = self.__get_state(row, col - 1) if (col > 0) else state
                    elif (action == 1):
                        to_state = self.__get_state(row + 1, col) if (row < 15) else state
                    elif (action == 2):
                        to_state = self.__get_state(row, col + 1) if (col < 15) else state
                    else:
                        to_state = self.__get_state(row - 1, col) if (row > 0) else state
                    reward = 1.0 if (to_state == self.finish_state and state != self.finish_state) else 0.0
                    to_state = to_state if (state != self.finish_state) else state
                    terminated = True if (to_state == self.finish_state) else False
                    probability = 1
                    self.P[state][action].append((probability, to_state , reward, terminated))
    
    def __get_state(self, row, col):
        return int(settings.COLS * row + col)

    def __get_coordinates(self, state):
        row = np.floor(state/settings.COLS);
        col = (state - settings.COLS*row)
        return row, col

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None:
            if not isinstance(options, dict):
                raise RuntimeError("Variable options is not a dictionary")
            self.delay = options.get('delay', 0.5)

        np.random.seed(seed)
        self.action = 0
        self.reward = 0.0
        self.state = np.random.randint(0, settings.NUM_TILES)
        self.current_battery = settings.BATTERY_LOAD
        self.world.reset(self.state, self.action)
        
        return self.state, {}

    def step(self, action):
        row, col = self.__get_coordinates(self.state)
        prev_state = self.state

        if np.random.random() < 1 - self.current_battery / settings.BATTERY_LOAD:
            # Elegir quedarse en la misma posición o ir a una diferente a la elegida por la acción
            neighbors = [(row, col)]
            expected_state = self.__get_coordinates(self.P[self.state][action][0][1])

            if (row > 0):
                neighbors.append((row - 1, col))
            if (col > 0):
                neighbors.append((row, col - 1))
            if (row < settings.COLS - 1):
                neighbors.append((row + 1, col))
            if (col < settings.ROWS - 1):
                neighbors.append((row, col + 1))
            
            neighbors.remove(expected_state)

            random_index = np.random.randint(0, len(neighbors))
            to_state = self.__get_state(neighbors[random_index][0], neighbors[random_index][1])
            
            if (neighbors[random_index][0] == row - 1):
                action = 3
            elif (neighbors[random_index][1] == col - 1):
                action = 0
            elif (neighbors[random_index][0] == row + 1):
                action = 1
            elif (neighbors[random_index][0] == col + 1):
                action = 2

            # print("should go to: {}, but goes to: {}".format(self.state, to_state))
            self.state = to_state
        else:
            # Ir a la dirección esperada por la acción
            self.state = self.P[self.state][action][0][1]
            pass
        
        self.action = action
        self.reward = self.P[prev_state][self.action][0][2]
        terminated = self.P[prev_state][self.action][0][3]

        self.world.update(
            self.state,
            self.action,
            self.reward,
            terminated
        )

        self.current_battery -= 1
        self.render()
        time.sleep(self.delay)
            
        if (terminated):
            self.win += 1
        elif (self.current_battery == 0):
            self.lose +=1

        return self.state, self.reward, terminated, self.current_battery == 0, {}

    def render(self):
        self.world.render(self.current_battery/settings.BATTERY_LOAD, (self.win, self.lose))

    def close(self):
        self.world.close()