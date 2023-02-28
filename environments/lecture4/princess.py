import time

import numpy as np

import pygame

import gym
from gym import spaces

from game.Game import Game


class PrincessEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, **kwargs):
        super().__init__()
        self.render_mode = kwargs.get("render_mode")
        self.game = Game("Princess Puzzle Env", self.render_mode)
        self.n = self.game.world.tile_map.rows * self.game.world.tile_map.cols
        self.observation_space = spaces.Discrete(self.n * self.n * self.n)
        self.action_space = spaces.Discrete(4)
        self.current_state = self.game.get_state()
        self.current_action = 0
        self.current_reward = 0.0
        self.delay = 1
        self.__build_P()

    def __build_P(self):
        self.P = { state : { action: [] for action in range(self.action_space.n) } for state in range(self.observation_space.n) } 
        self.terminal_state = None
        # simulate the game
        # (state, action) => (state', Reward)
        # s2 follows mc, s1 reflects mc
        for mc in range(self.n):
            for s1 in range(self.n):
                for s2 in range(self.n):                                                               
                    for action in range(self.action_space.n):
                        state = self.__compute_state_result(mc, s1, s2)
                        if (state == self.terminal_state):
                            self.P[state][action].append((1.0, state, 0.0, True))
                        else:
                            to_state, reward, terminated = self.__simulate_step(mc, s1, s2, action)
                            self.P[state][action].append((1.0, to_state, reward, terminated))
    
    def __simulate_step(self, mc: int, s1: int, s2: int, action: int):
        state = self.__compute_state_result(mc, s1, s2)
        mc_from = self.__get_coordinates(mc)
        s1_from = self.__get_coordinates(s1)
        s2_from = self.__get_coordinates(s2)
        if (action == 0):
            off_set = (0 , -1)
        elif (action == 1):
            off_set = (1 , 0)
        elif (action == 2):
            off_set = (0 , 1)
        else:
            off_set = (-1 , 0)

        # apply action to mc
        to_i, to_j = mc_from[0] + off_set[0], mc_from[1] + off_set[1]
        if (
            0 <= to_i < self.game.world.tile_map.rows
            and 0 <= to_j < self.game.world.tile_map.cols
            # and self.game.world.tile_map.tiles[to_i][to_j].busy_by != "ST"
            and (to_i, to_j) != s1_from
            and (to_i, to_j) != s2_from
            and self.game.world.tile_map.map[to_i][to_j] != 0
        ):
            # self.tile_map.tiles[i][j].busy_by = None
            # self.tile_map.tiles[n_i][n_j].busy_by = self.busy_mark
            # self.x, self.y = TileMap.to_screen(n_i, n_j)
            mc_to = (to_i, to_j)
        else:
            mc_to = mc_from
        
        # apply action to s1 based on mc movement
        to_i, to_j = s1_from[0] + off_set[0]*-1, s1_from[1] + off_set[1]*-1
        if (
            0 <= to_i < self.game.world.tile_map.rows
            and 0 <= to_j < self.game.world.tile_map.cols
            # and self.game.world.tile_map.tiles[to_i][to_j].busy_by != "ST"
            and (to_i, to_j) != s2_from
            and self.game.world.tile_map.map[to_i][to_j] != 0
        ):
            s1_to = (to_i, to_j)
        else:
            s1_to = s1_from
        
        # apply action to s2 based on mc movement
        to_i, to_j = s2_from[0] + off_set[0], s2_from[1] + off_set[1]
        if (
            0 <= to_i < self.game.world.tile_map.rows
            and 0 <= to_j < self.game.world.tile_map.cols
            # and self.game.world.tile_map.tiles[to_i][to_j].busy_by != "ST"
            and (to_i, to_j) != s1_to
            and self.game.world.tile_map.map[to_i][to_j] != 0
        ):
            s2_to = (to_i, to_j)
        else:
            s2_to = s2_from

        if s1_to == s2_to:
            # self.statue_1.undo_movement()
            s1_to = s1_from
            # self.statue_2.undo_movement()
            s2_to = s2_from

        to_state = self.__compute_state_result(
            self.__get_state(*mc_to), 
            self.__get_state(*s1_to), 
            self.__get_state(*s2_to)
            )
        
        assert to_state <= self.observation_space.n

        reward = 0.0
        terminated = False
        
        # check lost
        if (mc_to == s1_to or mc_to == s2_to):
            reward = -100.0
            terminated = True
        # check win
        elif (s1_to == self.game.world.target_1 and s2_to == self.game.world.target_2):
            reward = 1000.0
            terminated = True
            self.terminal_state = to_state
        # check win
        elif (s2_to == self.game.world.target_1 and s1_to == self.game.world.target_2):
            reward = 1000.0
            terminated = True
            self.terminal_state = to_state
        # state = state'
        elif (state == to_state):
            reward = -10.0
        # state != state'
        else:
            reward = -1.0

        return (to_state, reward, terminated)
    
    def __get_state(self, row: int, col: int):
        return int(self.game.world.tile_map.cols * row + col)

    def __get_coordinates(self, state: int):
        row = state // self.game.world.tile_map.cols
        col = (state - self.game.world.tile_map.cols * row)
        return row, col
   
    def __compute_state_result(self, mc, s1, s2):
        return mc * self.n**2 + s1 * self.n + s2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None:
            if not isinstance(options, dict):
                raise RuntimeError("Variable options is not a dictionary")
            self.delay = options.get("delay", 0.5)

        np.random.seed(seed)

        self.current_state = self.game.reset()
        self.current_action = 0
        self.current_reward = 0

        return self.__compute_state_result(*self.current_state), {}

    def step(self, action):
        self.current_action = action

        old_state = self.current_state
        self.current_state = self.game.update(self.current_action)

        terminated = False
        self.current_reward = -1.0
        
        if old_state == self.current_state:
            self.current_reward = -10.0
        elif self.game.world.check_lost():
            terminated = True
            self.current_reward = -100.0
        elif self.game.world.check_win():
            terminated = True
            self.current_reward = 1000.0

        if self.render_mode is not None:
            self.render()
            time.sleep(self.delay)

        _, to_state , reward, terminated2 = self.P[self.__compute_state_result(*old_state)][action][0]

        assert to_state == self.__compute_state_result(*self.current_state)
        assert reward == self.current_reward
        assert terminated == terminated2

        return (
            # self.__compute_state_result(*self.current_state),
            to_state,
            # self.current_reward,
            reward,
            # terminated,
            terminated2,
            False,
            {},
        )

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
