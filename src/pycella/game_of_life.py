"""
game_of_life
"""

from typing import Literal

import numpy as np
from scipy.signal import convolve2d


class GameOfLife:
    """
    Game of Life

    states: dead, alive

    initial grid: randomized

    state transitions
    - alive < 2 alive neighbors --> dead
    - alive > 3 alive neighbors --> dead
    - dead == 3 alive neighbors --> alive
    """

    states = {"dead": 0, "alive": 1}

    def __init__(
        self,
        rows: int,
        cols: int,
        prob_init: float = 0.5,
        neighborhood: np.ndarray = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        boundary: Literal["fill", "wrap", "symm"] = "fill",
    ):
        self.neighborhood = neighborhood
        self.boundary = boundary

        # init grid
        self.grid = np.random.rand(rows, cols) < prob_init
        self.grid = self.grid.astype(np.uint8)

    def next(self):
        # next states for this iteration
        alive = np.equal(self.grid, self.states["alive"])
        dead = np.equal(self.grid, self.states["dead"])

        # calc alive neighbors
        neighbors = convolve2d(
            alive, self.neighborhood, mode="same", boundary=self.boundary, fillvalue=0
        )

        # alive < 2 alive neighbors --> dead
        st2 = neighbors < 2
        self.grid[np.logical_and(alive, st2)] = self.states["dead"]

        # alive > 3 alive neighbors --> dead
        gt3 = neighbors > 3
        self.grid[np.logical_and(alive, gt3)] = self.states["dead"]

        # dead == 3 alive neighbors --> alive
        eq3 = neighbors == 3
        self.grid[np.logical_and(dead, eq3)] = self.states["alive"]
