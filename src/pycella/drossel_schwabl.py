"""
drossel_schwabl
"""
from collections import deque
from typing import Literal

import numpy as np
from scipy.signal import convolve2d


class DrosselSchwabl:
    """
    Implementation of the Drossel-Schwabl forest fire model.

    states: free, tree, fire

    initial grid: free

    state transitions
    - fire --> free
    - >= 1 fire neighbor --> fire
    - tree probability f --> fire
    - free probability p --> tree
    """

    states = {"free": 0, "tree": 1, "fire": 2}

    def __init__(
        self,
        rows: int,
        cols: int,
        prob_tree: float,
        prob_fire: float,
        neighborhood: np.ndarray = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        boundary: Literal["fill", "wrap", "symm"] = "fill",
    ):
        self.prob_fire = prob_fire
        self.prob_tree = prob_tree
        self.neighborhood = neighborhood
        self.boundary = boundary

        # init grid
        self.grid = np.zeros(shape=(rows, cols), dtype=np.uint8)

    def next(self):
        # initial states for this iteration
        free = np.equal(self.grid, self.states["free"])
        tree = np.equal(self.grid, self.states["tree"])
        fire = np.equal(self.grid, self.states["fire"])

        # collect the fire statistics
        fire_sum = fire.sum()
        self.collect_fire_stats(fire_sum)

        # --- state changes ---
        # fire --> free
        self.grid[fire] = self.states["free"]

        # tree with >= 1 neighbors on fire --> fire
        neighbor_on_fire = (
            convolve2d(
                fire,
                self.neighborhood,
                mode="same",
                boundary=self.boundary,
                fillvalue=0,
            )
            > 0
        )
        self.grid[np.logical_and(tree, neighbor_on_fire)] = self.states["fire"]

        # only grow new trees and start fires when there is no fire going on right now
        if fire_sum == 0:
            # tree probability --> fire
            new_fire = np.random.rand(*self.grid.shape) < self.prob_fire
            self.grid[np.logical_and(tree, new_fire)] = self.states["fire"]

            # free probability --> tree
            new_trees = np.random.rand(*self.grid.shape) < self.prob_tree
            self.grid[np.logical_and(free, new_trees)] = self.states["tree"]

    __curr = False
    """is there currently fire"""
    __fire_total = 0
    """total sum of trees on fire during this fire"""
    fire_stats = deque(maxlen=65536)
    """list of all total fire sums"""

    def collect_fire_stats(self, fire_sum: int) -> None:
        """
        count how many trees are on fire for the whole duration of the fire

        :remarks: cannot distinguish between multiple individual fires that start in the same iteration

        :param fire_sum: sum trees on fire currently
        """
        prev = self.__curr
        self.__curr = fire_sum > 0

        if self.__curr:
            # fire is still burning
            self.__fire_total = self.__fire_total + fire_sum
        elif not self.__curr and prev:
            # fire stopped
            self.fire_stats.append(self.__fire_total)
            self.__fire_total = 0
