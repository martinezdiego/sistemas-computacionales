from typing import List, Any, Tuple, Set, NoReturn
from enum import Enum
import random

from .MazeGenerator import MazeGenerator

class RecursiveDivisionMazeGenerator(MazeGenerator):
    
    DIRECTION = Enum('Direction', ['HORIZONTAL', 'VERTICAL'])

    def __init__(self,
        num_rows: int,
        num_cols: int,
        neighborhood: List[Tuple[int, int]] = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ) -> None:
        super().__init__(num_rows, num_cols, neighborhood)

    def generate(self, start: int = 0) -> None:
        self.__divide(0, 0, self.num_cols, self.num_rows, self.__choose_direction(self.num_cols, self.num_rows))
    
    def __choose_direction(self, width: int, height: int) -> DIRECTION:
        if (width < height):
            return self.DIRECTION.HORIZONTAL
        elif (height < width):
            return self.DIRECTION.VERTICAL
        else:
            num = random.random() * 100
            return self.DIRECTION.HORIZONTAL if (num % 2 == 0) else self.DIRECTION.VERTICAL

    def __divide(self, x: int, y: int, width: int, height: int, direction: DIRECTION) -> None:
        if (width < 2 or height < 2):
            return
        
        horizontal = (direction == self.DIRECTION.HORIZONTAL)

        # where will the wall be drawn from?
        wx = x + (0 if horizontal else random.randint(0, width - 2))
        wy = y + (random.randint(0, height - 2) if horizontal else 0)

        # where will the passage through the wall exist?
        px = wx + (random.randint(0, width) if horizontal else 0)
        py = wy + (0 if horizontal else random.randint(0, width))

        # how long will the wall be?
        length = width if horizontal else height

        # what direction will the wall be drawn?
        dx = 1 if horizontal else 0
        dy = 0 if horizontal else 1

        # draw the wall
        for _ in range(length):
            self.grid[wy][wx] |= direction.value if (wx != px or wy != py) else 0
            wx += dx
            wy += dy

        # determine the bounds of the subfields
        nx, ny = x, y
        w, h = (width, (wy - y + 1)) if horizontal else ((wx - x + 1), height)

        self.__divide(nx, ny, w, h, self.__choose_direction(w, h))

        nx, ny = (x, (wy + 1)) if horizontal else ((wx + 1), y)
        w, h = (width, (y + height - wy - 1)) if horizontal else ((x + width - wx - 1), height)
        
        self.__divide(nx, ny, w, h, self.__choose_direction(w, h))
    
    def render(self) -> None :
        print(" " + "_" * (self.num_cols * 2 - 1))
        for row in range(self.num_rows):
            print("|", end="")
            for col in range(self.num_cols):
                bottom = row + 1 >= self.num_rows
                bottom1  = (self.grid[row][col] & self.DIRECTION.HORIZONTAL.value != 0 or bottom)
                bottom2 = (col + 1 < self.num_cols and self.grid[row][col + 1] & self.DIRECTION.HORIZONTAL.value != 0 or bottom)
                right   = (self.grid[row][col] & self.DIRECTION.VERTICAL.value != 0 or col + 1 >= self.num_cols)

                print("_" if bottom else " ", end="")
                print("|" if right else "_" if (bottom1 and bottom2) else " ", end="")
            print("")

