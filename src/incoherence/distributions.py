from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generic, TypeVar

import numpy as np  # type: ignore

X = TypeVar("X")
Bool = int  # a slight hack not to bother with True/False


# Representing discrete probability distributions using dictionaries
# Generic over a type X representing the domain of the distribution
@dataclass
class P(Generic[X]):
    dist: Dict[X, float]

    def sample(self):
        """Samples a value from the distribution."""
        return np.random.choice(np.array(list(self.dist.keys())), p=np.array(list(self.dist.values())))

    def expectation(self, f: Callable[[X], float] | None = None):
        """Computes the expectation of a function f under the distribution."""
        if f is None:
            f = lambda x: x  # type: ignore
        return sum(prob * f(value) for value, prob in self.dist.items() if prob != 0)  # type: ignore

    def __post_init__(self):
        # Normalize the dist to form a proper distribution
        total = sum(self.dist.values())
        dd = {value: prob / total for value, prob in self.dist.items()}
        self.dist = defaultdict(float, dd)

    def __eq__(self, other):
        # return self.dist == other.dist
        return self.__isclose__(other)

    def __sub__(self, other):
        return {k: v - other.dist[k] for k, v in self.dist.items()}

    def __isclose__(self, other):
        keys = list(self.dist.keys())
        return set(keys) == set(other.dist.keys()) and \
                np.allclose(np.array([self.dist[k] for k in keys]),
                           np.array([other.dist[k] for k in keys]))


def bernoulli(prob) -> P[Bool]:
    return P({1: prob, 0: 1 - prob})


def make_distribution_from_empirical(data):
    total = len(data)
    counts = defaultdict(int)
    for value in data:
        counts[tuple(value) if isinstance(value, list) else value] += 1
    return P({value: count / total for value, count in counts.items()})


__all__ = ["Bool", "P", "bernoulli", "make_distribution_from_empirical"]
