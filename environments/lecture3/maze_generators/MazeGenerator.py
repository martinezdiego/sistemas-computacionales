"""
Base for any maze generator
"""
from typing import List, Any, Tuple, Set, NoReturn


class MazeGenerator:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        neighborhood: List[Tuple[int, int]] = [(0, -1), (1, 0), (0, 1), (-1, 0)],
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.neighborhood = neighborhood
        self.walls: Set[Tuple[int, int]] = set()
        self.grid: List[List[int]] = [ [0 for _ in range(self.num_cols)] for _ in range(self.num_rows) ]
    
    def get_grid(self) -> List[List[int]]:
        return self.grid

    def generate(self, start: int = 0) -> NoReturn:
        raise NotImplementedError()

    def render(self) -> NoReturn:
        raise NotImplementedError()
