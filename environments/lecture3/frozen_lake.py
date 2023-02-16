"""
Frozen Lake environment as a maze
"""
from typing import List, Tuple
import numpy as np
import time

import gym
import pygame

import maze_generators
import settings
from tilemap import TileMap

class FrozenLake(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, **kwargs):
        self._rows = kwargs.get("rows", 4)
        self._cols = kwargs.get("cols", 4)
        self.render_mode = kwargs.get("render_mode")
        
        # maze init
        self.maze_generator = kwargs.get(
            "maze_generator_class", maze_generators.RecursiveDivisionMazeGenerator
        )(self._rows, self._cols)
        self.maze_generator.generate()
        self.grid = self.maze_generator.get_grid()

        # env constants
        self.NUM_TILES = self._rows * self._cols
        self.NUM_ACTIONS = 4
        self.VIRTUAL_WIDTH = settings.TILE_SIZE * self._cols
        self.VIRTUAL_HEIGHT = settings.TILE_SIZE * self._rows
        self.WINDOW_WIDTH = self.VIRTUAL_WIDTH * settings.H_SCALE
        self.WINDOW_HEIGHT = self.VIRTUAL_HEIGHT * settings.V_SCALE
        
        # env init
        self.observation_space = gym.spaces.Discrete(self.NUM_TILES)
        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)
        
        self.connected_components = self.__get_connected_components()
        self.connected_components.sort(reverse=True, key=len)

        self.initial_state = self.connected_components[0][0]
        self.finish_state = max(self.connected_components[0])
        self.path = self.__build_path(self.initial_state, self.finish_state)
        holes_candidates = [ tile for tile in range(self.NUM_TILES) if (tile not in self.path) ]
        self.holes = np.random.choice(holes_candidates, size=(len(holes_candidates) * 33 // 100)).tolist()

        self.current_state = self.initial_state
        self.current_action = 1
        self.current_reward = 0.0
        self.delay = settings.DEFAULT_DELAY

        self.__init_P()

        # world init
        if self.render_mode is not None:
            self.init_render_mode(self.render_mode)

        self.render_character = True
        self.render_goal = True
        self.tilemap = None
        self.__create_tilemap()

    def init_render_mode(self, render_mode):
        self.render_mode = render_mode

        pygame.init()
        pygame.display.init()
        pygame.mixer.music.play(loops=-1)
        self.render_surface = pygame.Surface(
            (self.VIRTUAL_WIDTH, self.VIRTUAL_HEIGHT)
        )
        self.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Frozen Lake Environment")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None:
            if not isinstance(options, dict):
                raise RuntimeError("Variable options is not a dictionary")
            self.delay = options.get('delay', 0.5)
        
        self.current_state = self.initial_state
        self.current_action = 1
        self.current_reward = 0.0
        self.render_character = True
        self.render_goal = True

        for tile in self.tilemap.tiles:
            if tile.texture_name == "cracked_hole":
                tile.texture_name = "hole"

        return self.current_state, {}

    def step(self, action):
        _, to_state , reward, terminated = self.P[self.current_state][action][0]

        self.current_state = to_state
        self.current_action = action

        if (self.render_mode is not None):
            if terminated:
                if to_state == self.finish_state:
                    self.render_goal = False
                    settings.SOUNDS["win"].play()
                else:
                    self.tilemap.tiles[to_state].texture_name = "cracked_hole"
                    self.render_character = False
                    settings.SOUNDS["ice_cracking"].play()
                    settings.SOUNDS["water_splash"].play()
            
            self.render()
            time.sleep(self.delay)

        return to_state, reward, terminated, False, {}

    def render(self):        
        self.render_surface.fill((0, 0, 0))

        self.tilemap.render(self.render_surface)

        self.render_surface.blit(
            settings.TEXTURES["stool"],
            (self.tilemap.tiles[self.initial_state].x, self.tilemap.tiles[self.initial_state].y),
        )

        if self.render_goal:
            self.render_surface.blit(
                settings.TEXTURES["goal"],
                (
                    self.tilemap.tiles[self.finish_state].x,
                    self.tilemap.tiles[self.finish_state].y,
                ),
            )

        if self.render_character:
            self.render_surface.blit(
                settings.TEXTURES["character"][self.current_action],
                (self.tilemap.tiles[self.current_state].x, self.tilemap.tiles[self.current_state].y),
            )

        self.__render_walls()
        
        self.screen.blit(
            pygame.transform.scale(self.render_surface, self.screen.get_size()), (0, 0)
        )
        
        pygame.event.pump()
        pygame.display.update()
    
    def close(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        pygame.display.quit()
        pygame.quit()
    
    def __init_P(self) -> None:
        self.P = { state : { action: [] for action in range(self.NUM_ACTIONS) } for state in range(self.NUM_TILES) }
        for row in range(self._rows):
            for col in range(self._cols):
                state = self.__get_state(row, col)
                for action in range(self.NUM_ACTIONS):
                    if (action == 0):
                        to_state = self.__get_state(row, col - 1) if (col > 0) else state
                        if (to_state != state):
                            _, vertical_wall = self.__get_walls(row, col - 1)
                            if (vertical_wall):
                                to_state = state
                    elif (action == 1):
                        to_state = self.__get_state(row + 1, col) if (row < self._rows - 1) else state
                        if (to_state != state):
                            horizontal_wall, _ = self.__get_walls(row, col)
                            if (horizontal_wall):
                                to_state = state
                    elif (action == 2):
                        to_state = self.__get_state(row, col + 1) if (col < self._cols - 1) else state
                        if (to_state != state):
                            _, vertical_wall = self.__get_walls(row, col)
                            if (vertical_wall):
                                to_state = state
                    else:
                        to_state = self.__get_state(row - 1, col) if (row > 0) else state
                        if (to_state != state):
                            horizontal_wall, _ = self.__get_walls(row - 1, col)
                            if (horizontal_wall):
                                to_state = state

                    if (state == self.finish_state):
                        self.P[state][action].append((1.0, state, 1.0, True))
                    else:
                        reward = 1.0 if (to_state == self.finish_state) else 0.0
                        terminated = True if (to_state == self.finish_state or (to_state in self.holes)) else False
                        self.P[state][action].append((1.0, to_state , reward, terminated))

    def __get_state(self, row: int, col: int) -> int:
        return int(self._cols * row + col)

    def __get_coordinates(self, state: int) -> Tuple[int, int]:
        row = np.floor(state/self._cols);
        col = (state - self._cols*row)
        return int(row.item()), int(col.item())
    
    def __create_tilemap(self) -> None:
        tile_texture_names = ["ice" for _ in range(self.NUM_TILES)]
        for _, actions_table in self.P.items():
            for _, possibilities in actions_table.items():
                for _, state, reward, terminated in possibilities:
                    if terminated:
                        if reward > 0:
                            self.finish_state = state
                        else:
                            tile_texture_names[state] = "hole"

        tile_texture_names[self.finish_state] = "ice"
        self.tilemap = TileMap(self._rows, self._cols, tile_texture_names)

    def __get_walls(self, x: int, y: int) -> Tuple[bool, bool]:
        horizontal = self.grid[x][y] & self.maze_generator.DIRECTION.HORIZONTAL.value != 0
        vertical = self.grid[x][y] & self.maze_generator.DIRECTION.VERTICAL.value != 0
        return (horizontal, vertical)

    def __get_connected_components(self) -> List[List[int]]:
        components = []
        current_component = []
        visited = [False for _ in range(self.NUM_TILES)]

        for state in range(self.NUM_TILES):
            if (not visited[state]):
                self.__dfs(state, visited, current_component)
                components.append(current_component)
                current_component = []

        return components

    def __dfs(self, state: int, visited: List[bool], current_component: List[int]) -> None:
        current_component.append(state)
        visited[state] = True
        x, y = self.__get_coordinates(state)
        neighbors = []

        if (y > 0):
            # check if there is a path to left neighbor
            _, vertical_wall = self.__get_walls(x, y - 1)
            if (not vertical_wall):
                neighbors.append(self.__get_state(x, y - 1))

        if (x < self._rows - 1):
            # check if there is a path to bottom neighbor
            horizontal_wall, _ = self.__get_walls(x, y)
            if (not horizontal_wall):
                neighbors.append(self.__get_state(x + 1, y))

        if (y < self._cols - 1):
            # check if there is a path to right neighbor
            _, vertical_wall = self.__get_walls(x, y)
            if (not vertical_wall):
                neighbors.append(self.__get_state(x, y + 1))
        
        if (x > 0):
            # check if there is a path to up neighbor
            horizontal_wall, _ = self.__get_walls(x - 1, y)
            if (not horizontal_wall):
                neighbors.append(self.__get_state(x - 1, y))

        for neighbor in neighbors:
            if (not visited[neighbor]):
                self.__dfs(neighbor, visited, current_component)

    def __build_path(self, from_state: int, to_state: int) -> List[int]:
        parent = [ -1 for _ in range(self.NUM_TILES) ]
        queue = [ from_state ]
        visited = [False for _ in range(self.NUM_TILES)]
        
        parent[from_state] = from_state
        path = []

        while (len(queue) > 0):
            current_state = queue.pop(0)
            visited[current_state] = True
            
            if (current_state == to_state):
                break

            x, y = self.__get_coordinates(current_state)
            neighbors = []

            if (y > 0):
                # check if there is a path to left neighbor
                _, vertical_wall = self.__get_walls(x, y - 1)
                if (not vertical_wall):
                    neighbors.append(self.__get_state(x, y - 1))

            if (x < self._rows - 1):
                # check if there is a path to bottom neighbor
                horizontal_wall, _ = self.__get_walls(x, y)
                if (not horizontal_wall):
                    neighbors.append(self.__get_state(x + 1, y))

            if (y < self._cols - 1):
                # check if there is a path to right neighbor
                _, vertical_wall = self.__get_walls(x, y)
                if (not vertical_wall):
                    neighbors.append(self.__get_state(x, y + 1))
            
            if (x > 0):
                # check if there is a path to up neighbor
                horizontal_wall, _ = self.__get_walls(x - 1, y)
                if (not horizontal_wall):
                    neighbors.append(self.__get_state(x - 1, y))

            for neighbor in neighbors:
                if (not visited[neighbor]):
                    parent[neighbor] = current_state
                    queue.append(neighbor)
        
        path = [ to_state ]
        current_state = to_state

        while (parent[current_state] != current_state):
            current_state = parent[current_state]
            path.append(current_state)
        
        return path
    
    def __render_walls(self):
        for tile in range(self.NUM_TILES):
            row, col = self.__get_coordinates(tile)
            bottom  = (self.grid[row][col] & self.maze_generator.DIRECTION.HORIZONTAL.value != 0)
            right   = (self.grid[row][col] & self.maze_generator.DIRECTION.VERTICAL.value != 0)
            x, y = col * settings.TILE_SIZE, row * settings.TILE_SIZE
           
            if (bottom):
                # render bottom wall
                start_pos = (x, y + settings.TILE_SIZE)
                end_pos = (x + settings.TILE_SIZE, y + settings.TILE_SIZE)
                pygame.draw.line(self.render_surface, pygame.Color(0, 0, 0), start_pos , end_pos)
            if (right):
                # render right wall
                start_pos = (x + settings.TILE_SIZE, y)
                end_pos = (x + settings.TILE_SIZE, y + settings.TILE_SIZE)
                pygame.draw.line(self.render_surface, pygame.Color(0, 0, 0), start_pos , end_pos)
