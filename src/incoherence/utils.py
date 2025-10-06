from collections import defaultdict

import numpy as np  # type: ignore

from .distributions import P
from .mdp import MDP, Policy, Trajectory


def flatten(ll):
    return [x for l in ll for x in l]


def print_trajectory(trajectory: Trajectory, prob=None, with_actions=False):
    """Prints a trajectory in a human-readable format."""

    reward = trajectory[-1][2]
    if with_actions:
        s = [[f"{state}", f" --{action}--> "] for state, action, _ in trajectory]
        del s[-1][-1]
        s = "".join(flatten(s))
        print(s, end="")
    else:
        states = [state for state, _, _ in trajectory]
        print(" -> ".join(states), end="")
    print(f" ({reward})", end="")
    if prob is None:
        print()
    else:
        print(f" - Prob: {prob}")


def sort_dict(d):
    return {key: value for key, value in sorted(d.items())}


def print_policy(mdp: MDP, policy: Policy):
    """Prints the policy in a human-readable format."""
    policy = sort_dict(policy)
    for state, actions in policy.items():
        # if mdp.state_time[state] == mdp.time_horizon-1:
        #     continue
        aps = sort_dict(actions.dist)
        rounded = {a: float(round(p, 8)) for a, p in aps.items()}
        print(f"Policy({state}) = {rounded}")


def print_rewards_mean(rewards):
    rs = defaultdict(list)
    for state, actions_probs in rewards.items():
        for action, prob in actions_probs.items():
            if True:  # action == " ":
                rs[state].append(prob.dist[1])
    print("Rewards: ", {k: float(np.mean(v)) for k, v in rs.items()})


def print_rewards(rewards):
    """Prints the rewards in a human-readable format."""
    rewards = sort_dict(rewards)
    for state, actions in rewards.items():
        rounded = {a: round(p.dist[1], 5) for a, p in actions.items()}
        rounded = sort_dict(rounded)
        print(f"Reward({state}) = {rounded}")


def print_js(js):
    print([round(j, 8) for j in js])


def print_trajectory_prob(prob):
    prob = sort_dict(prob.dist)
    for traj, p in prob.items():
        if p > 0:
            print_trajectory(traj, p)


__all__ = [
    "flatten",
    "print_js",
    "print_policy",
    "print_rewards",
    "print_rewards_mean",
    "print_trajectory",
    "print_trajectory_prob",
    "sort_dict",
]
