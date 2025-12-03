"""
host_pathogen
"""

from typing import Literal

import numpy as np
from scipy.signal import convolve2d


class HostPathogen:
    """
    Implementation of a host-pathogen model.

    states: empty, healthy, infected

    initial grid: random empty and healthy

    state transitions
    - empty and healthy neighbors and probability -> healthy
    - healthy and infected neighbors and probability * num_infected -> infected
    - infected -> empty
    """

    states = {"empty": 0, "healthy": 1, "infected": 2}

    def __init__(
            self,
            rows: int,
            cols: int,
            prob_reproduce: float,
            prob_infect: float,
            neighborhood: np.ndarray = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            boundary: Literal["fill", "wrap", "symm"] = "fill",
    ):
        self.rows = rows
        self.cols = cols
        self.prob_reproduce = prob_reproduce
        self.prob_infect = prob_infect
        self.neighborhood = neighborhood
        self.boundary = boundary

        # init grid randomly with states empty, healthy
        self.grid = np.random.rand(rows, cols) * 3
        self.grid = self.grid.astype(np.uint8)

    def next(self):
        # initial states for this iteration
        empty = np.equal(self.grid, self.states["empty"])
        healthy = np.equal(self.grid, self.states["healthy"])
        infected = np.equal(self.grid, self.states["infected"])

        self.collect_statistics(empty, healthy, infected)

        # --- state changes ---
        # empty and healthy neighbors and probability -> healthy
        random_grid = np.random.rand(self.rows, self.cols)
        healthy_neighbors = (
            convolve2d(
                healthy,
                self.neighborhood,
                mode="same",
                boundary=self.boundary,
                fillvalue=0,
            )

        )
        reproduce = random_grid < (healthy_neighbors * self.prob_reproduce)
        self.grid[np.logical_and(empty, reproduce)] = self.states["healthy"]

        # healthy and infected neighbors and probability * num_infected -> infected
        random_grid = np.random.rand(self.rows, self.cols)
        infected_neighbors = (
            convolve2d(
                infected,
                self.neighborhood,
                mode="same",
                boundary=self.boundary,
                fillvalue=0,
            )
        )
        infect_neighbor = random_grid < (infected_neighbors * self.prob_infect)
        self.grid[np.logical_and(healthy, infect_neighbor)] = self.states["infected"]

        # infected -> empty
        self.grid[infected] = self.states["empty"]

    def has_next(self) -> bool:
        return not np.all(self.grid == self.states["healthy"]) and not np.all(self.grid == self.states["empty"])

    stats_empty = []
    stats_healthy = []
    stats_infected = []

    def collect_statistics(self, empty:np.ndarray, healthy:np.ndarray, infected:np.ndarray)->None:
        self.stats_empty.append(empty.sum())
        self.stats_healthy.append(healthy.sum())
        self.stats_infected.append(infected.sum())
