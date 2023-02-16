"""
Maze generator based on depth first search algorithm
"""
from typing import List, Any, Tuple, Set

import random

from .MazeGenerator import MazeGenerator


class DepthFirstMazeGenerator(MazeGenerator):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        neighborhood: List[Tuple[int, int]] = [(0, -1), (1, 0), (0, 1), (-1, 0)],
    ) -> None:
        super().__init__(num_rows, num_cols, neighborhood)

    def _init_walls(self) -> None:
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                current_index = i * self.num_cols + j
                for offset_i, offset_j in self.neighborhood:
                    n_i, n_j = i + offset_i, j + offset_j

                    if not ((0 <= n_i < self.num_rows) and (0 <= n_j < self.num_cols)):
                        continue

                    neighbor_index = n_i * self.num_cols + n_j

                    if (current_index, neighbor_index) in self.walls or (
                        neighbor_index,
                        current_index,
                    ) in self.walls:
                        continue

                    self.walls.add((current_index, neighbor_index))

    def generate(self, start: int = 0) -> None:
        self._init_walls()

        stack: List[int] = [start]
        visited: Set[int] = {start}

        while len(stack) > 0:
            current_index = stack.pop()

            unvisited_neighbors: List[int] = []

            i, j = current_index // self.num_cols, current_index % self.num_cols

            for offset_i, offset_j in self.neighborhood:
                n_i, n_j = i + offset_i, j + offset_j
                if not ((0 <= n_i < self.num_rows) and (0 <= n_j < self.num_cols)):
                    continue
                neighbor_index = n_i * self.num_cols + n_j
                if neighbor_index not in visited:
                    unvisited_neighbors.append(neighbor_index)

            if len(unvisited_neighbors) == 0:
                continue

            stack.append(current_index)

            neighbor = random.choice(unvisited_neighbors)

            # Remove the wall
            if (current_index, neighbor) in self.walls:
                self.walls.remove((current_index, neighbor))
            if (neighbor, current_index) in self.walls:
                self.walls.remove((neighbor, current_index))

            stack.append(neighbor)
            visited.add(neighbor)
    
    def render(self) -> None:
        # render the top wall
        print("-" * int(self.num_cols * 2 + 1))

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # evaluate if there is a left wall
                current_index = i * self.num_cols + j
                left_index = i * self.num_cols + j - 1
                has_left_wall = (
                    j == 0
                    or (current_index, left_index) in self.walls
                    or (left_index, current_index) in self.walls
                )

                # render the left wall if exists
                if has_left_wall:
                    # every left cell has a left wall
                    print("|", end="")
                else:
                    # space if there is no a left wall
                    print(" ", end="")

                # render the cell
                print(" ", end="")

            # render the right wall for the current row
            print("|")

            # render the bottom wall when if exists
            for j in range(self.num_cols * 2 + 1):
                current_index = i * self.num_cols + j
                bottom_index = (i + 1) * self.num_cols + j
                has_bottom_wall = (
                    i == self.num_rows - 1
                    or j % 2 == 0
                    or (current_index, bottom_index) in self.walls
                    or (bottom_index, current_index) in self.walls
                )
                if has_bottom_wall:
                    print("-", end="")
                else:
                    # space if there is not a bottom wall
                    print(" ", end="")

            # finally, end of line
            print("")
