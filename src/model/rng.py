"""
Generator Bundle Class for Deterministic Simulations
"""

from numpy import random

class RngBundle:
    """
    Stream of generators
    """

    def __init__(self, seed: int):
        init, activation, action, tiebreak, order = random.SeedSequence(seed).spawn(5)
        self.init = random.default_rng(init)
        self.activation = random.default_rng(activation)
        self.action = random.default_rng(action)
        self.tiebreak = random.default_rng(tiebreak)
        self.order = random.default_rng(order)
